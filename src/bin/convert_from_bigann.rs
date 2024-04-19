use ann_dataset::{AnnDataset, Hdf5File, InMemoryAnnDataset, Metric, PointSet, QuerySet};
use ann_dataset_converter::util::{get_largest, new_progress_bar};
use clap::Parser;
use linfa_linalg::norm::Norm;
use ndarray::{Array1, Array2, Axis, Zip};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path to an hdf5 file containing the data.
    #[clap(long, required = true)]
    path: String,

    /// Label of data points.
    #[clap(long, required = true)]
    data_points: String,

    /// Label of train query points.
    #[clap(long)]
    train_query_points: Option<String>,

    /// Label of validation query points.
    #[clap(long)]
    validation_query_points: Option<String>,

    /// Label of test query points.
    #[clap(long)]
    test_query_points: Option<String>,

    /// Top-k nearest neighbors to add as ground truth.
    #[clap(long, required = true)]
    top_k: usize,

    /// Path to the output file where an `AnnDataset` object will be stored.
    #[clap(long, required = true)]
    output: String,
}

fn read_data(path: &str, label: &str) -> anyhow::Result<Array2<f32>> {
    let file = hdf5::File::open(path)?;
    let expected_file = file.dataset(label).unwrap();
    let num_points: usize = expected_file.shape()[0];
    let num_dimensions: usize = expected_file.shape()[1];

    let vectors = expected_file.read_raw::<f32>().unwrap();
    let data = Array2::from_shape_vec((num_points, num_dimensions), vectors)?;

    Ok(data)
}

fn find_gts(
    data: &Array2<f32>,
    queries: &Array2<f32>,
    k: usize,
) -> (Array2<usize>, Array2<usize>, Array2<usize>) {
    let mut gt_euclidean = Array2::<usize>::zeros((queries.nrows(), k));
    let mut gt_cosine = Array2::<usize>::zeros((queries.nrows(), k));
    let mut gt_ip = Array2::<usize>::zeros((queries.nrows(), k));

    let norms = Array1::from(
        data.outer_iter()
            .map(|point| point.norm_l2())
            .collect::<Vec<_>>(),
    );

    let pb = new_progress_bar("Finding ground truth", 1, queries.nrows());
    Zip::from(queries.axis_iter(Axis(0)))
        .and(gt_euclidean.axis_iter_mut(Axis(0)))
        .and(gt_cosine.axis_iter_mut(Axis(0)))
        .and(gt_ip.axis_iter_mut(Axis(0)))
        .par_for_each(|query, mut gt_euclidean, mut gt_cosine, mut gt_ip| {
            let scores = data.dot(&query);
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
        dataset.get_data_points().get_dense().unwrap(),
        query_set.get_points().get_dense().unwrap(),
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

    let dense = read_data(args.path.as_str(), args.data_points.as_str())
        .expect("Unable to read data points.");
    let data_points =
        PointSet::new(Some(dense), None).expect("Failed to create a point set from data points.");

    let mut dataset = InMemoryAnnDataset::create(data_points);

    if let Some(train) = args.train_query_points {
        println!("Processing train query points...");
        let dense = read_data(args.path.as_str(), train.as_str())
            .unwrap_or_else(|_| panic!("Failed to read query points labeled '{}'", train));
        let query_points = PointSet::new(Some(dense), None)
            .unwrap_or_else(|_| panic!("Failed to create query point set '{}'", train));
        let mut query_set = QuerySet::new(query_points);

        attach_gt(&dataset, &mut query_set, args.top_k);
        dataset.add_train_query_set(query_set);
    }

    if let Some(validation) = args.validation_query_points {
        println!("Processing validation query points...");
        let dense = read_data(args.path.as_str(), validation.as_str())
            .unwrap_or_else(|_| panic!("Failed to read query points labeled '{}'", validation));
        let query_points = PointSet::new(Some(dense), None)
            .unwrap_or_else(|_| panic!("Failed to create query point set '{}'", validation));
        let mut query_set = QuerySet::new(query_points);

        attach_gt(&dataset, &mut query_set, args.top_k);
        dataset.add_validation_query_set(query_set);
    }

    if let Some(test) = args.test_query_points {
        println!("Processing test query points...");
        let dense = read_data(args.path.as_str(), test.as_str())
            .unwrap_or_else(|_| panic!("Failed to read query points labeled '{}'", test));
        let query_points = PointSet::new(Some(dense), None)
            .unwrap_or_else(|_| panic!("Failed to create query point set '{}'", test));
        let mut query_set = QuerySet::new(query_points);

        attach_gt(&dataset, &mut query_set, args.top_k);
        dataset.add_test_query_set(query_set);
    }

    dataset
        .write(args.output.as_str())
        .expect("Failed to write the dataset into output file.");
    println!("Dataset created and serialized:\n{}", dataset);
}
