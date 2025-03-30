use ann_dataset::{AnnDataset, Hdf5File, InMemoryAnnDataset, Metric, PointSet, QuerySet};
use ann_dataset_converter::util::{get_largest, new_progress_bar};
use anyhow::anyhow;
use clap::Parser;
use flate2::read::GzDecoder;
use ndarray::{Array1, Array2, Axis, Zip};
use serde::Deserialize;
use sprs::{CsMat, TriMat};
use std::cmp::max;
use std::fs::File;
use std::io::BufReader;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Comma-separated list of paths to the (gzipped) collection of sparse vectors in json format.
    ///
    /// One convenient way to prepare this list in bash is as follows, assuming
    /// shards are in a directory called `vectors/`:
    /// $ ... --data-points `ls vectors/*json.gz | tr '\n' ',' | sed 's/,*$//g'`
    #[clap(
        long,
        use_value_delimiter = true,
        value_delimiter = ',',
        required = true
    )]
    data_points: Vec<String>,

    /// Path to train query points.
    #[clap(long)]
    train_query_points: Option<String>,

    /// Path to validation query points.
    #[clap(long)]
    validation_query_points: Option<String>,

    /// Path to test query points.
    #[clap(long)]
    test_query_points: Option<String>,

    /// Top-k nearest neighbors to add as ground truth.
    #[clap(long, required = true)]
    top_k: usize,

    /// Path to the output file where an `AnnDataset` object will be stored.
    #[clap(long, required = true)]
    output: String,
}

#[derive(Debug, Deserialize)]
struct JsonCollection {
    vectors: Vec<JsonVector>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
#[allow(dead_code)]
enum JsonVector {
    Single {
        id: String,
        coordinates: usize,
        values: f32,
    },
    Multi {
        id: String,
        coordinates: Vec<usize>,
        values: Vec<f32>,
    },
    SingleInt {
        id: u32,
        coordinates: usize,
        values: f32,
    },
    MultiInt {
        id: u32,
        coordinates: Vec<usize>,
        values: Vec<f32>,
    },
}

impl JsonVector {
    fn get_coordinates_values(&self) -> (Vec<usize>, Vec<f32>) {
        match self {
            JsonVector::Single {
                id: _,
                coordinates,
                values,
            } => (vec![*coordinates], vec![*values]),
            JsonVector::Multi {
                id: _,
                coordinates,
                values,
            } => {
                let mut indices = (0..coordinates.len()).collect::<Vec<usize>>();
                indices.sort_by_key(|&i| coordinates[i]);

                (
                    indices.iter().map(|&i| coordinates[i]).collect(),
                    indices.iter().map(|&i| values[i]).collect(),
                )
            }
            JsonVector::SingleInt {
                id: _,
                coordinates,
                values,
            } => (vec![*coordinates], vec![*values]),
            JsonVector::MultiInt {
                id: _,
                coordinates,
                values,
            } => {
                let mut indices = (0..coordinates.len()).collect::<Vec<usize>>();
                indices.sort_by_key(|&i| coordinates[i]);

                (
                    indices.iter().map(|&i| coordinates[i]).collect(),
                    indices.iter().map(|&i| values[i]).collect(),
                )
            }
        }
    }
}

fn read_data(paths: &[String], max_dimension: Option<usize>) -> anyhow::Result<CsMat<f32>> {
    let mut triplets_ids: Vec<usize> = vec![];
    let mut triplets_coordinates: Vec<usize> = vec![];
    let mut triplets_values: Vec<f32> = vec![];

    let mut id = 0_usize;
    let mut num_dimensions = 0_usize;

    for path in paths {
        println!("Reading data from {}", path);
        let f = File::open(path).unwrap_or_else(|_| panic!("Unable to open {}", path));
        let json_collection: JsonCollection = if path.ends_with("gz") {
            let reader = BufReader::new(GzDecoder::new(f));
            serde_json::from_reader(reader)?
        } else {
            let reader = BufReader::new(f);
            serde_json::from_reader(reader)?
        };

        json_collection.vectors.iter().try_for_each(|vector| {
            let (coordinates, values) = vector.get_coordinates_values();
            if coordinates.len() != values.len() {
                return Err(anyhow!(format!(
                    "Vector with id {} has {} coordinates but {} values",
                    id,
                    coordinates.len(),
                    values.len()
                )));
            }

            coordinates.iter().enumerate().for_each(|(i, &coordinate)| {
                if let Some(max_dimension) = max_dimension {
                    if coordinate >= max_dimension {
                        return;
                    }
                }
                triplets_ids.push(id);
                triplets_coordinates.push(coordinate);
                triplets_values.push(values[i]);
            });

            num_dimensions = max(num_dimensions, *coordinates.iter().max().unwrap_or(&0) + 1);
            id += 1;
            anyhow::Ok(())
        })?;
    }

    let sparse = TriMat::from_triplets(
        (id, max_dimension.unwrap_or(num_dimensions)),
        triplets_ids,
        triplets_coordinates,
        triplets_values,
    );
    let sparse: CsMat<_> = sparse.to_csr();
    Ok(sparse)
}

fn find_gts(
    data: &CsMat<f32>,
    queries: &CsMat<f32>,
    k: usize,
) -> (Array2<usize>, Array2<usize>, Array2<usize>) {
    let mut gt_euclidean = Array2::<usize>::zeros((queries.rows(), k));
    let mut gt_cosine = Array2::<usize>::zeros((queries.rows(), k));
    let mut gt_ip = Array2::<usize>::zeros((queries.rows(), k));

    let norms = Array1::from(
        data.outer_iterator()
            .map(|point| point.l2_norm())
            .collect::<Vec<_>>(),
    );

    let queries = queries.outer_iterator().collect::<Vec<_>>();

    let pb = new_progress_bar("Finding ground truth", 1, queries.len());
    Zip::indexed(gt_euclidean.axis_iter_mut(Axis(0)))
        .and(gt_cosine.axis_iter_mut(Axis(0)))
        .and(gt_ip.axis_iter_mut(Axis(0)))
        .par_for_each(|query_id, mut gt_euclidean, mut gt_cosine, mut gt_ip| {
            let query = queries[query_id];
            let scores = Array1::from((data * &query).to_dense().to_vec());
            gt_ip.assign(&get_largest(scores.view(), k));
            gt_cosine.assign(&get_largest((&scores / &norms).view(), k));
            gt_euclidean.assign(&get_largest((-&norms * &norms + 2_f32 * &scores).view(), k));
            pb.inc(1);
        });

    pb.finish_and_clear();

    (gt_euclidean, gt_cosine, gt_ip)
}

fn attach_gt(dataset: &InMemoryAnnDataset<f32>, query_set: &mut QuerySet<f32>, top_k: usize) {
    let (gt_euclidean, gt_cosine, gt_ip) = find_gts(
        dataset.get_data_points().get_sparse().unwrap(),
        query_set.get_points().get_sparse().unwrap(),
        top_k,
    );
    query_set
        .add_ground_truth(Metric::InnerProduct, gt_ip)
        .expect("Failed to add ground-truth to the query set");

    query_set
        .add_ground_truth(Metric::Cosine, gt_cosine)
        .expect("Failed to add ground-truth to the query set");

    query_set
        .add_ground_truth(Metric::Euclidean, gt_euclidean)
        .expect("Failed to add ground-truth to the query set");
}

fn main() {
    let args = Args::parse();

    let sparse = read_data(&args.data_points, None).expect("Unable to read data points.");
    let num_dimensions = sparse.cols();
    let data_points =
        PointSet::new(None, Some(sparse)).expect("Failed to create a point set from data points.");

    let mut dataset = InMemoryAnnDataset::create(data_points);

    if let Some(train) = args.train_query_points {
        println!("Processing train query points...");
        let sparse = read_data(&[train.clone()], Some(num_dimensions))
            .unwrap_or_else(|_| panic!("Failed to read query points labeled '{}'", &train));
        let query_points = PointSet::new(None, Some(sparse))
            .unwrap_or_else(|_| panic!("Failed to create query point set '{}'", &train));
        let mut query_set = QuerySet::new(query_points);

        attach_gt(&dataset, &mut query_set, args.top_k);
        dataset.add_train_query_set(query_set);
    }

    if let Some(validation) = args.validation_query_points {
        println!("Processing validation query points...");
        let sparse = read_data(&[validation.clone()], Some(num_dimensions))
            .unwrap_or_else(|_| panic!("Failed to read query points labeled '{}'", &validation));
        let query_points = PointSet::new(None, Some(sparse))
            .unwrap_or_else(|_| panic!("Failed to create query point set '{}'", &validation));
        let mut query_set = QuerySet::new(query_points);

        attach_gt(&dataset, &mut query_set, args.top_k);
        dataset.add_validation_query_set(query_set);
    }

    if let Some(test) = args.test_query_points {
        println!("Processing test query points...");
        let sparse = read_data(&[test.clone()], Some(num_dimensions))
            .unwrap_or_else(|_| panic!("Failed to read query points labeled '{}'", &test));
        let query_points = PointSet::new(None, Some(sparse))
            .unwrap_or_else(|_| panic!("Failed to create query point set '{}'", &test));
        let mut query_set = QuerySet::new(query_points);

        attach_gt(&dataset, &mut query_set, args.top_k);
        dataset.add_test_query_set(query_set);
    }

    dataset
        .write(args.output.as_str())
        .expect("Failed to write the dataset into output file.");
    println!("Dataset created and serialized:\n{}", dataset);
}
