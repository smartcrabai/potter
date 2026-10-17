//! Deformation bind state and parameter parity for Surface Deform, Mesh Deform, Laplacian Deform,
//! and the Corrective Smooth bind data, including chained deforms.
//!
//! Surface Deform and Laplacian Deform positions were produced by Blender 5.2.2 (headless) on the
//! same fixtures. Mesh Deform expectations are the Blender-measured invariants: a uniform cage
//! translation moves each vertex by the translation times its group weight, and moving the cage or
//! owner object after binding has no effect.

use std::{
    error::Error,
    fs,
    path::Path,
    process::{Command, Output},
};

use glam::DVec3;
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    model::{Id, SceneDoc},
};
use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult = Result<(), Box<dyn Error>>;

const SURFACE_TOLERANCE: f64 = 2.0e-5;
const LAPLACIAN_TOLERANCE: f64 = 1.0e-4;
const MESH_TOLERANCE: f64 = 1.0e-5;

const SUBJECT_LOCATION: [f64; 3] = [0.2, -0.1, 0.5];
const SUBJECT_ROTATION: [f64; 4] = [
    0.267_261_241_912_424_4,
    0.534_522_483_824_848_8,
    0.801_783_725_737_273_2,
    0.0,
];
const SUBJECT_SCALE: [f64; 3] = [1.0, 0.8, 1.2];
const TARGET_LOCATION: [f64; 3] = [-0.4, 0.3, 0.2];
const TARGET_ROTATION: [f64; 4] = [0.0, 0.0, -0.247_403_959_254_522_9, 0.968_912_421_710_644_7];
const TARGET_SCALE: [f64; 3] = [1.3, 0.7, 1.1];
const TARGET_MOVED_LOCATION: [f64; 3] = [0.1, 0.05, 0.5];
const TARGET_MOVED_ROTATION: [f64; 4] =
    [0.0, 0.0, 0.382_683_432_365_089_8, 0.923_879_532_511_286_7];
const TARGET_DELTAS: [(usize, [f64; 3]); 4] = [
    (0, [0.0, 0.0, 0.035]),
    (1, [0.018, 0.0, -0.012]),
    (2, [0.0, -0.016, 0.024]),
    (3, [-0.011, 0.008, -0.019]),
];
const CAGE_TRANSLATION: [f64; 3] = [0.3, -0.2, 0.5];
const BLENDER_GRID_EDGES: [[usize; 2]; 40] = [
    [0, 5],
    [0, 1],
    [1, 6],
    [5, 6],
    [1, 2],
    [2, 7],
    [6, 7],
    [2, 3],
    [3, 8],
    [7, 8],
    [3, 4],
    [4, 9],
    [8, 9],
    [5, 10],
    [6, 11],
    [10, 11],
    [7, 12],
    [11, 12],
    [8, 13],
    [12, 13],
    [9, 14],
    [13, 14],
    [10, 15],
    [11, 16],
    [15, 16],
    [12, 17],
    [16, 17],
    [13, 18],
    [17, 18],
    [14, 19],
    [18, 19],
    [15, 20],
    [16, 21],
    [20, 21],
    [17, 22],
    [21, 22],
    [18, 23],
    [22, 23],
    [19, 24],
    [23, 24],
];

const SURFACE_DEFAULT: [[f64; 3]; 25] = [
    [-1.000_000_119, -0.999_999_821, -0.248_274_416],
    [-0.499_999_970, -1.000_000_000, -0.225_197_464],
    [-0.000_000_030, -0.999_999_881, -0.099_941_403],
    [0.499_999_940, -0.999_999_940, 0.025_314_709],
    [0.999_999_881, -1.000_000_000, 0.048_391_655],
    [-1.000_000_000, -0.499_999_970, -0.112_059_832],
    [-0.500_000_000, -0.499_999_970, -0.076_482_914],
    [0.000_000_000, -0.500_000_000, 0.061_273_109],
    [0.499_999_940, -0.499_999_940, 0.199_029_163],
    [1.000_000_000, -0.499_999_970, 0.234_606_087],
    [-1.000_000_000, 0.000_000_000, -0.048_332_900],
    [-0.499_999_970, -0.000_000_015, -0.000_256_091],
    [-0.000_000_004, 0.000_000_014, 0.149_999_976],
    [0.499_999_970, -0.000_000_051, 0.300_256_044],
    [1.000_000_000, -0.000_000_060, 0.348_332_971],
    [-1.000_000_119, 0.500_000_060, -0.162_059_769],
    [-0.500_000_417, 0.500_000_179, -0.101_482_846],
    [-0.000_085_619, 0.500_144_899, 0.061_303_847],
    [0.503_811_836, 0.507_666_528, 0.222_910_672],
    [1.010_961_294, 0.509_083_688, 0.278_834_611],
    [-0.998_930_335, 0.999_674_678, -0.348_586_559],
    [-0.498_427_778, 0.998_802_006, -0.275_387_555],
    [0.002_112_350, 1.006_417_155, -0.100_004_546],
    [0.504_332_244, 1.022_444_606, 0.074_970_193],
    [1.016_770_959, 1.031_681_538, 0.143_233_642],
];

const SURFACE_FALLOFF_GROUP: [[f64; 3]; 25] = [
    [-1.000_000_000, -1.000_000_000, -0.248_274_371],
    [-0.500_000_000, -1.000_000_000, -0.225_197_479],
    [-0.000_000_018, -0.999_999_940, -0.099_941_410],
    [0.500_000_000, -1.000_000_000, 0.025_314_681],
    [0.999_999_940, -1.000_000_000, 0.048_391_592],
    [-0.999_999_940, -0.499_999_940, -0.112_059_832],
    [-0.500_000_000, -0.500_000_000, -0.076_482_959],
    [0.000_000_000, -0.500_000_000, 0.061_273_113],
    [0.499_999_970, -0.499_999_970, 0.199_029_177],
    [1.000_000_000, -0.500_000_000, 0.234_606_072],
    [-1.000_000_000, 0.000_000_000, -0.048_332_941],
    [-0.499_999_970, -0.000_000_009, -0.000_256_087],
    [-0.000_000_001, 0.000_000_002, 0.150_000_006],
    [0.500_000_000, -0.000_000_014, 0.300_256_073],
    [1.000_000_000, -0.000_000_036, 0.348_332_942],
    [-1.000_000_000, 0.500_000_000, -0.162_059_829],
    [-0.500_003_338, 0.500_001_848, -0.101_482_272],
    [-0.000_077_665, 0.500_140_011, 0.061_301_228],
    [0.500_457_108, 0.500_795_543, 0.223_891_467],
    [1.003_879_905, 0.503_499_985, 0.282_578_498],
    [-0.999_428_213, 0.999_826_074, -0.348_441_243],
    [-0.499_811_947, 0.999_856_710, -0.275_220_215],
    [0.000_760_893, 1.002_310_634, -0.099_964_201],
    [0.503_028_810, 1.013_453_603, 0.075_001_098],
    [1.001_869_559, 1.004_112_601, 0.147_844_166],
];

const SURFACE_SPARSE: [[f64; 3]; 25] = [
    [-1.000_000_000, -1.000_000_000, -0.248_274_371],
    [-0.500_000_000, -1.000_000_000, -0.225_197_479],
    [-0.000_000_017, -0.999_999_940, -0.099_941_410],
    [0.499_999_970, -0.999_999_940, 0.025_314_702],
    [1.000_000_000, -1.000_000_000, 0.048_391_558],
    [-0.999_999_940, -0.499_999_970, -0.112_059_839],
    [-0.500_000_000, -0.500_000_000, -0.076_482_944],
    [0.000_000_000, -0.499_999_970, 0.061_273_109],
    [0.500_000_000, -0.500_000_000, 0.199_029_192],
    [0.999_999_940, -0.499_999_940, 0.234_606_072],
    [-1.000_000_000, 0.000_000_000, -0.048_332_945],
    [-0.499_999_970, -0.000_000_006, -0.000_256_085],
    [0.000_000_000, 0.000_000_000, 0.150_000_006],
    [0.499_999_970, -0.000_000_042, 0.300_256_073],
    [1.000_000_000, -0.000_000_054, 0.348_332_942],
    [-1.000_000_000, 0.500_000_000, -0.162_059_829],
    [-0.500_000_000, 0.500_000_000, -0.101_482_965],
    [-0.000_025_802, 0.500_043_333, 0.061_282_363],
    [0.502_674_222, 0.506_610_215, 0.223_280_236],
    [1.009_895_802, 0.508_068_979, 0.279_388_577],
    [-1.000_000_000, 1.000_000_000, -0.348_274_380],
    [-0.499_327_719, 0.999_487_698, -0.275_278_747],
    [0.001_235_721, 1.003_754_139, -0.099_978_343],
    [0.503_032_625, 1.016_670_823, 0.075_104_713],
    [1.000_000_000, 1.000_000_000, 0.148_391_560],
];

const SURFACE_INVERT: [[f64; 3]; 25] = [
    [-1.000_000_000, -0.999_999_881, -0.248_274_401],
    [-0.499_999_970, -1.000_000_000, -0.225_197_479],
    [0.000_000_000, -1.000_000_000, -0.099_941_403],
    [0.499_999_970, -1.000_000_000, 0.025_314_704],
    [0.999_999_940, -1.000_000_000, 0.048_391_595],
    [-1.000_000_000, -0.500_000_000, -0.112_059_847],
    [-0.500_000_000, -0.499_999_970, -0.076_482_929],
    [0.000_000_000, -0.500_000_000, 0.061_273_109],
    [0.500_000_000, -0.500_000_000, 0.199_029_192],
    [0.999_999_940, -0.499_999_940, 0.234_606_087],
    [-1.000_000_000, 0.000_000_000, -0.048_332_941],
    [-0.500_000_000, 0.000_000_000, -0.000_256_081],
    [-0.000_000_004, 0.000_000_011, 0.149_999_976],
    [0.500_000_000, -0.000_000_018, 0.300_256_073],
    [1.000_000_000, 0.000_000_000, 0.348_332_971],
    [-1.000_000_119, 0.500_000_060, -0.162_059_784],
    [-0.500_001_311, 0.500_000_715, -0.101_482_689],
    [0.000_000_000, 0.500_000_000, 0.061_273_117],
    [0.503_061_891, 0.505_586_445, 0.223_114_163],
    [1.004_350_305, 0.503_752_351, 0.282_323_420],
    [-1.000_000_000, 1.000_000_000, -0.348_274_380],
    [-0.498_743_713, 0.999_042_749, -0.275_349_349],
    [0.000_845_022, 1.002_566_934, -0.099_966_675],
    [0.500_000_000, 1.000_000_000, 0.075_314_678],
    [1.012_795_806, 1.026_695_848, 0.144_576_117],
];

const SURFACE_CHAINED: [[f64; 3]; 25] = [
    [-1.000_000_000, -0.999_999_762, -0.248_274_416],
    [-0.499_999_940, -1.000_000_000, -0.225_197_464],
    [-0.000_000_030, -0.999_999_881, -0.099_941_388],
    [0.499_999_940, -0.999_999_881, 0.025_314_685],
    [0.999_999_940, -1.000_000_000, 0.048_391_655],
    [-0.999_999_881, -0.499_999_940, -0.112_059_832],
    [-0.500_000_000, -0.499_999_940, -0.076_482_885],
    [0.000_000_000, -0.500_000_000, 0.061_273_105],
    [0.499_999_940, -0.499_999_940, 0.199_029_177],
    [1.000_000_000, -0.500_000_000, 0.234_606_117],
    [-1.000_000_000, 0.000_000_000, -0.048_332_870],
    [-0.500_000_000, -0.000_000_022, -0.000_256_076],
    [-0.000_000_005, 0.000_000_022, 0.149_999_961],
    [0.499_999_940, -0.000_000_069, 0.300_256_044],
    [1.000_000_000, -0.000_000_060, 0.348_332_971],
    [-1.000_000_238, 0.500_000_119, -0.162_059_754],
    [-0.500_001_788, 0.500_000_954, -0.101_482_525],
    [-0.000_100_406, 0.500_172_496, 0.061_309_226],
    [0.503_819_585, 0.507_324_815, 0.222_898_036],
    [1.010_918_498, 0.509_232_283, 0.278_867_066],
    [-0.998_966_813, 0.999_685_764, -0.348_575_920],
    [-0.498_428_702, 0.998_802_781, -0.275_387_436],
    [0.002_112_437, 1.006_417_274, -0.100_004_613],
    [0.504_523_218, 1.022_438_765, 0.074_922_636],
    [1.016_382_933, 1.032_525_659, 0.143_427_968],
];

const LAPLACIAN_BORDER_ITERATIONS_3: [[f64; 3]; 25] = [
    [-0.962_602_556, -1.010_230_660, 0.378_446_519],
    [-0.455_155_492, -1.020_076_036, 0.432_966_530],
    [0.050_362_136, -1.021_329_999, 0.468_472_689],
    [0.542_759_538, -1.021_855_831, 0.458_822_817],
    [1.039_118_409, -1.016_809_225, 0.398_531_556],
    [-0.949_688_315, -0.513_273_656, 0.433_677_137],
    [-0.450_488_985, -0.520_104_766, 0.458_757_460],
    [0.049_812_917, -0.522_060_454, 0.478_633_612],
    [0.546_177_804, -0.522_890_031, 0.475_435_257],
    [1.042_777_896, -0.522_862_494, 0.446_842_164],
    [-0.950_206_935, -0.020_067_114, 0.468_752_354],
    [-0.450_363_070, -0.021_789_888, 0.478_724_420],
    [0.050_283_052, -0.022_632_083, 0.492_568_016],
    [0.550_176_740, -0.023_372_583, 0.495_592_475],
    [1.051_192_403, -0.025_260_454, 0.489_051_968],
    [-0.951_727_629, 0.476_578_593, 0.459_142_596],
    [-0.451_742_917, 0.476_954_371, 0.475_609_243],
    [0.049_820_788, 0.477_466_971, 0.495_704_174],
    [0.550_422_728, 0.477_511_138, 0.501_291_394],
    [1.048_423_171, 0.478_734_940, 0.492_642_641],
    [-0.941_845_059, 0.980_397_582, 0.398_959_875],
    [-0.451_838_851, 0.976_792_634, 0.447_196_215],
    [0.049_832_959, 0.978_310_049, 0.489_413_291],
    [0.550_407_648, 0.978_127_301, 0.492_817_372],
    [1.038_190_722, 0.982_824_326, 0.444_264_412],
];

const LAPLACIAN_INTERIOR_INVERT: [[f64; 3]; 25] = [
    [-0.958_089_411, -1.015_850_425, 0.506_027_877],
    [-0.454_937_190, -1.017_725_229, 0.484_991_729],
    [0.049_623_404, -1.018_995_404, 0.457_452_953],
    [0.555_069_089, -1.018_325_686, 0.425_366_133],
    [1.057_611_227, -1.017_662_525, 0.404_932_201],
    [-0.954_402_626, -0.515_427_589, 0.484_905_183],
    [-0.453_130_156, -0.517_412_663, 0.469_902_426],
    [0.049_750_134, -0.518_606_186, 0.448_435_068],
    [0.553_323_269, -0.517_659_903, 0.422_862_589],
    [1.054_753_065, -0.515_958_607, 0.407_223_731],
    [-0.949_228_406, -0.014_930_387, 0.457_237_840],
    [-0.449_368_894, -0.017_579_850, 0.448_358_923],
    [0.049_630_415, -0.018_272_044, 0.432_499_856],
    [0.549_712_002, -0.017_446_764, 0.419_227_540],
    [1.049_684_048, -0.014_630_899, 0.409_688_234],
    [-0.945_945_919, 0.484_292_716, 0.425_167_799],
    [-0.447_116_762, 0.482_461_810, 0.422_760_308],
    [0.050_560_001, 0.481_545_657, 0.419_192_880],
    [0.546_640_038, 0.482_969_880, 0.416_760_534],
    [1.044_965_982, 0.485_659_033, 0.412_739_545],
    [-0.943_410_337, 0.983_105_540, 0.404_726_386],
    [-0.445_490_777, 0.981_909_990, 0.407_103_002],
    [0.050_698_020, 0.981_150_150, 0.409_642_160],
    [0.544_844_925, 0.982_504_308, 0.412_732_035],
    [1.041_645_288, 0.984_405_160, 0.412_075_460],
];

const LAPLACIAN_BORDER_ITERATIONS_1: [[f64; 3]; 25] = [
    [-0.960_990_131, -1.013_216_734, 0.380_200_803],
    [-0.452_735_722, -1.020_231_128, 0.427_765_638],
    [0.051_025_029, -1.020_626_903, 0.461_828_411],
    [0.542_703_390, -1.021_567_225, 0.459_042_609],
    [1.040_146_112, -1.017_656_326, 0.410_211_474],
    [-0.949_912_190, -0.516_165_614, 0.428_017_288],
    [-0.450_328_499, -0.520_356_774, 0.448_653_758],
    [0.049_593_769, -0.521_468_222, 0.467_457_414],
    [0.545_805_216, -0.522_483_289, 0.468_409_628],
    [1.042_858_481, -0.523_405_969, 0.449_426_919],
    [-0.950_973_392, -0.020_844_094, 0.461_866_409],
    [-0.450_979_739, -0.021_490_179, 0.467_474_699],
    [0.049_640_723, -0.021_917_911, 0.479_574_144],
    [0.549_567_997, -0.022_763_496, 0.485_013_515],
    [1.050_696_373, -0.025_038_136, 0.484_243_244],
    [-0.952_994_466, 0.477_129_608, 0.459_127_069],
    [-0.452_706_724, 0.477_789_760, 0.468_468_010],
    [0.049_087_580, 0.478_307_724, 0.485_064_268],
    [0.549_989_641, 0.478_179_365, 0.492_588_133],
    [1.048_330_069, 0.478_991_836, 0.490_403_414],
    [-0.944_242_299, 0.982_314_229, 0.410_344_929],
    [-0.453_246_534, 0.978_283_763, 0.449_567_020],
    [0.049_075_015, 0.979_295_492, 0.484_419_435],
    [0.550_497_293, 0.978_973_269, 0.490_446_925],
    [1.039_763_093, 0.983_763_993, 0.453_088_582],
];

fn subject_xy(index: usize) -> (f64, f64) {
    (
        -1.0 + 0.5 * (index % 5) as f64,
        -1.0 + 0.5 * (index / 5) as f64,
    )
}

fn subject_height(x: f64, y: f64) -> f64 {
    0.2 * (1.7 * x).sin() + 0.15 * (2.3 * y).cos() + 0.05 * x * y
}

fn target_height(x: f64, y: f64) -> f64 {
    0.3 * (1.1 * x).sin() + 0.2 * y * y - 0.1 * x * y + 0.05 * x
}

fn wobbled_subject(index: usize) -> [f64; 3] {
    let (x, y) = subject_xy(index);
    [x, y, subject_height(x, y)]
}

fn flat_subject(index: usize) -> [f64; 3] {
    let (x, y) = subject_xy(index);
    [x, y, 0.0]
}

fn init(scene: &Path) -> TestResult {
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(scene)
        .arg("--json")
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "scene initialization failed: {}",
            String::from_utf8_lossy(&output.stdout)
        )
        .into());
    }
    Ok(())
}

fn apply(scene: &Path, operations: &Value) -> Result<Output, Box<dyn Error>> {
    let batch = scene.with_extension("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": operations,
        }))?,
    )?;
    Ok(Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(batch)
        .arg("--json")
        .output()?)
}

fn applied(output: &Output) -> TestResult {
    if !output.status.success() {
        return Err(format!(
            "apply failed (status={}): stdout={} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(())
}

fn load_doc(scene: &Path) -> Result<SceneDoc, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(
        scene.join("scene.json"),
    )?)?)
}

fn evaluated_subject(scene: &Path) -> Result<Vec<DVec3>, Box<dyn Error>> {
    let doc = load_doc(scene)?;
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
    let mesh = snapshot
        .meshes
        .get(&Id::new("subject")?)
        .ok_or("evaluated subject mesh is missing")?;
    Ok(mesh.vertices.iter().map(|vertex| vertex.co).collect())
}
fn evaluated_laplacian_subject(scene: &Path) -> Result<Vec<DVec3>, Box<dyn Error>> {
    let mut doc = load_doc(scene)?;
    let subject_id = Id::new("subject")?;
    let data_id = doc
        .nodes
        .get(&subject_id)
        .and_then(|node| node.data.as_ref())
        .cloned()
        .ok_or("subject data is missing")?;
    {
        let mesh = doc
            .data_blocks
            .get_mut(&data_id)
            .and_then(|data| data.mesh.as_mut())
            .ok_or("subject mesh is missing")?;
        let mut reordered_edges = Vec::with_capacity(mesh.edges.len());
        for [first, second] in BLENDER_GRID_EDGES {
            let first_id = mesh
                .vertices
                .get(first)
                .ok_or("Blender edge vertex is outside the mesh")?
                .id;
            let second_id = mesh
                .vertices
                .get(second)
                .ok_or("Blender edge vertex is outside the mesh")?
                .id;
            let edge_index = mesh
                .edges
                .iter()
                .position(|edge| {
                    (edge.vertices[0] == first_id && edge.vertices[1] == second_id)
                        || (edge.vertices[0] == second_id && edge.vertices[1] == first_id)
                })
                .ok_or("Blender edge does not exist in the Potter mesh")?;
            reordered_edges.push(mesh.edges.remove(edge_index));
        }
        if !mesh.edges.is_empty() {
            return Err("Blender and Potter edge counts differ".into());
        }
        mesh.edges = reordered_edges;
    }
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
    let mesh = snapshot
        .meshes
        .get(&subject_id)
        .ok_or("evaluated subject mesh is missing")?;
    Ok(mesh.vertices.iter().map(|vertex| vertex.co).collect())
}

fn assert_close(
    actual: &[DVec3],
    expected: &[[f64; 3]],
    tolerance: f64,
    label: &str,
) -> TestResult {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label}: {} vertices, expected {}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    let mut worst = (0_usize, 0.0_f64);
    for (index, (position, reference)) in actual.iter().zip(expected).enumerate() {
        let error = position.distance(DVec3::from_array(*reference));
        if error > worst.1 {
            worst = (index, error);
        }
    }
    if worst.1 > tolerance {
        return Err(format!(
            "{label}: vertex {} off by {} (tolerance {tolerance}); actual {:?}, expected {:?}",
            worst.0, worst.1, actual[worst.0], expected[worst.0]
        )
        .into());
    }
    Ok(())
}

fn vertex_move(node: &str, index: usize, translation: [f64; 3]) -> Value {
    json!({
        "op": "mesh.transform_elements",
        "target": {"id": node},
        "elements": {"domain": "vertex", "ids": [format!("v{index}")]},
        "translation": translation,
    })
}

fn bind(modifier: &str) -> Value {
    json!({"op": "modifier.bind", "target": {"id": "subject"}, "modifier_id": modifier})
}

fn unbind(modifier: &str) -> Value {
    json!({"op": "modifier.unbind", "target": {"id": "subject"}, "modifier_id": modifier})
}

fn group_operations(weights: &[f64]) -> Vec<Value> {
    let assignments = weights
        .iter()
        .enumerate()
        .map(|(index, weight)| json!({"vertex_id": index, "weight": weight}))
        .collect::<Vec<_>>();
    vec![
        json!({"op": "vertex_group.create", "target": {"id": "subject"}, "id": "influence", "name": "Influence"}),
        json!({"op": "vertex_group.assign", "target": {"id": "subject"}, "group_id": "influence", "weights": assignments}),
    ]
}

fn partial_weights() -> Vec<f64> {
    (0..25)
        .map(|index| 0.2 + 0.8 * ((index % 3) as f64 / 2.0))
        .collect()
}

fn sparse_weights() -> Vec<f64> {
    (0..25)
        .map(|index| {
            if index % 4 == 0 {
                0.0
            } else {
                0.3 + 0.7 * ((index % 5) as f64 / 4.0)
            }
        })
        .collect()
}

fn surface_operations(
    modifiers: &[(&str, Value)],
    group: Option<Vec<f64>>,
    operations_after_create: Vec<Value>,
    target_object_move: bool,
) -> Value {
    let mut operations = vec![json!({"op": "node.create", "id": "subject", "kind": "grid",
            "params": {"size": 2.0, "x_subdivisions": 4, "y_subdivisions": 4},
            "transform": {"translation": SUBJECT_LOCATION, "rotation": SUBJECT_ROTATION, "scale": SUBJECT_SCALE}})];
    for index in 0..25 {
        let (x, y) = subject_xy(index);
        operations.push(vertex_move(
            "subject",
            index,
            [0.0, 0.0, subject_height(x, y)],
        ));
    }
    operations.push(json!({"op": "node.create", "id": "target", "kind": "grid",
        "params": {"size": 3.0, "x_subdivisions": 3, "y_subdivisions": 3},
        "transform": {"translation": TARGET_LOCATION, "rotation": TARGET_ROTATION, "scale": TARGET_SCALE}}));
    for index in 0..16 {
        let x = -1.5 + (index % 4) as f64;
        let y = -1.5 + (index / 4) as f64;
        operations.push(vertex_move(
            "target",
            index,
            [0.0, 0.0, target_height(x, y)],
        ));
    }
    if let Some(weights) = group {
        operations.extend(group_operations(&weights));
    }
    for (id, params) in modifiers {
        operations.push(json!({"op": "modifier.create", "target": {"id": "subject"},
            "id": id, "type": "surface_deform", "params": params}));
    }
    operations.extend(operations_after_create);
    for (index, delta) in TARGET_DELTAS {
        operations.push(vertex_move("target", index, delta));
    }
    if target_object_move {
        operations.push(json!({"op": "node.update", "target": {"id": "target"},
            "set": {"transform": {"translation": TARGET_MOVED_LOCATION,
                "rotation": TARGET_MOVED_ROTATION, "scale": TARGET_SCALE}}}));
    }
    Value::Array(operations)
}

fn surface_params(falloff: f64, strength: f64) -> Value {
    json!({"target": "target", "falloff": falloff, "strength": strength})
}

fn build_scene(
    directory: &Path,
    name: &str,
    operations: &Value,
) -> Result<(Output, std::path::PathBuf), Box<dyn Error>> {
    let scene = directory.join(name);
    init(&scene)?;
    let output = apply(&scene, operations)?;
    Ok((output, scene))
}

#[test]
fn surface_deform_default_bind_matches_blender() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[("deform", surface_params(4.0, 1.0))],
        None,
        vec![bind("deform")],
        false,
    );
    let (output, scene) = build_scene(directory.path(), "default", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_subject(&scene)?,
        &SURFACE_DEFAULT,
        SURFACE_TOLERANCE,
        "surface default",
    )
}

#[test]
fn surface_deform_falloff_strength_and_group_match_blender() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[(
            "deform",
            json!({"target": "target", "falloff": 2.5, "strength": 0.6, "vertex_group": "Influence"}),
        )],
        Some(partial_weights()),
        vec![bind("deform")],
        false,
    );
    let (output, scene) = build_scene(directory.path(), "falloff_group", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_subject(&scene)?,
        &SURFACE_FALLOFF_GROUP,
        SURFACE_TOLERANCE,
        "surface falloff/strength/group",
    )
}

#[test]
fn surface_deform_sparse_bind_matches_blender() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[(
            "deform",
            json!({"target": "target", "falloff": 6.0, "strength": 0.9,
                "vertex_group": "Influence", "use_sparse_bind": true}),
        )],
        Some(sparse_weights()),
        vec![bind("deform")],
        false,
    );
    let (output, scene) = build_scene(directory.path(), "sparse", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_subject(&scene)?,
        &SURFACE_SPARSE,
        SURFACE_TOLERANCE,
        "surface sparse bind",
    )
}

#[test]
fn surface_deform_inverted_group_matches_blender() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[(
            "deform",
            json!({"target": "target", "falloff": 3.0, "strength": 1.0,
                "vertex_group": "Influence", "invert_vertex_group": true}),
        )],
        Some(partial_weights()),
        vec![bind("deform")],
        false,
    );
    let (output, scene) = build_scene(directory.path(), "invert", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_subject(&scene)?,
        &SURFACE_INVERT,
        SURFACE_TOLERANCE,
        "surface inverted group",
    )
}
#[test]
fn surface_deform_missing_vertex_group_acts_as_no_group() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[(
            "deform",
            json!({
                "target": "target",
                "falloff": 4.0,
                "strength": 1.0,
                "vertex_group": "Missing",
            }),
        )],
        None,
        vec![bind("deform")],
        false,
    );
    let (output, scene) = build_scene(directory.path(), "missing_surface_group", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_subject(&scene)?,
        &SURFACE_DEFAULT,
        SURFACE_TOLERANCE,
        "surface missing vertex group",
    )
}

#[test]
fn surface_deform_target_object_transform_is_frozen_at_bind() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[("deform", surface_params(4.0, 1.0))],
        None,
        vec![bind("deform")],
        true,
    );
    let (output, scene) = build_scene(directory.path(), "target_moved", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_subject(&scene)?,
        &SURFACE_DEFAULT,
        SURFACE_TOLERANCE,
        "surface target object moved after bind",
    )
}

#[test]
fn surface_deform_chained_binds_match_blender() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[
            ("first", surface_params(4.0, 1.0)),
            ("second", surface_params(3.0, 0.5)),
        ],
        None,
        vec![bind("first"), bind("second")],
        false,
    );
    let (output, scene) = build_scene(directory.path(), "chained", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_subject(&scene)?,
        &SURFACE_CHAINED,
        SURFACE_TOLERANCE,
        "surface chained",
    )
}

#[test]
fn surface_deform_sparse_bind_without_group_is_rejected() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[(
            "deform",
            json!({"target": "target", "falloff": 4.0, "strength": 1.0, "use_sparse_bind": true}),
        )],
        None,
        vec![bind("deform")],
        false,
    );
    let (output, _scene) = build_scene(directory.path(), "sparse_no_group", &operations)?;
    if output.status.success() {
        return Err("sparse surface bind without any bound vertex must fail".into());
    }
    Ok(())
}

#[test]
fn unbound_surface_deform_leaves_the_mesh_unchanged() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[("deform", surface_params(4.0, 1.0))],
        None,
        Vec::new(),
        false,
    );
    let (output, scene) = build_scene(directory.path(), "unbound", &operations)?;
    applied(&output)?;
    let rest = (0..25).map(wobbled_subject).collect::<Vec<_>>();
    assert_close(
        &evaluated_subject(&scene)?,
        &rest,
        SURFACE_TOLERANCE,
        "unbound surface deform",
    )
}

#[test]
fn surface_deform_rebind_after_unbind_matches_default_bind() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[("deform", surface_params(4.0, 1.0))],
        None,
        vec![bind("deform"), unbind("deform"), bind("deform")],
        false,
    );
    let (output, scene) = build_scene(directory.path(), "rebind", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_subject(&scene)?,
        &SURFACE_DEFAULT,
        SURFACE_TOLERANCE,
        "surface rebind after unbind",
    )
}

#[test]
fn surface_deform_second_bind_on_bound_modifier_is_rejected() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[("deform", surface_params(4.0, 1.0))],
        None,
        vec![bind("deform"), bind("deform")],
        false,
    );
    let (output, _scene) = build_scene(directory.path(), "double_bind", &operations)?;
    if output.status.success() {
        return Err("binding an already bound surface deform must fail".into());
    }
    Ok(())
}

fn mesh_operations(
    group: Option<Vec<f64>>,
    precision: u64,
    dynamic: bool,
    invert: bool,
    operations_after_bind: Vec<Value>,
) -> Value {
    let mut operations = vec![
        json!({"op": "node.create", "id": "subject", "kind": "grid",
            "params": {"size": 2.0, "x_subdivisions": 4, "y_subdivisions": 4}}),
        json!({"op": "node.create", "id": "cage", "kind": "box", "params": {"size": 4.0}}),
    ];
    let mut params = json!({"object": "cage", "precision": precision,
        "use_dynamic_bind": dynamic, "invert_vertex_group": invert});
    if let Some(weights) = group {
        operations.extend(group_operations(&weights));
        params["vertex_group"] = json!("Influence");
    }
    operations.push(json!({"op": "modifier.create", "target": {"id": "subject"},
        "id": "deform", "type": "mesh_deform", "params": params}));
    operations.push(bind("deform"));
    operations.extend(operations_after_bind);
    Value::Array(operations)
}

fn cage_translation_operations() -> Vec<Value> {
    (0..8)
        .map(|index| vertex_move("cage", index, CAGE_TRANSLATION))
        .collect()
}
fn asymmetric_mesh_operations(precision: u64, dynamic: bool, invert: bool) -> Value {
    let mut operations = vec![
        json!({"op": "node.create", "id": "subject", "kind": "grid",
            "params": {"size": 2.0, "x_subdivisions": 4, "y_subdivisions": 4},
            "transform": {"translation": [0.2, -0.1, 0.5],
                "rotation": [0.0, 0.0, 0.149_438_132_473_599_22, 0.988_771_077_936_042_2],
                "scale": [0.9, 1.2, 0.75]}}),
        json!({"op": "node.create", "id": "cage", "kind": "box",
            "params": {"size": 4.0},
            "transform": {"translation": [-0.4, 0.3, -0.2],
                "rotation": [0.0, 0.0, -0.099_833_416_646_828_15, 0.995_004_165_278_025_8],
                "scale": [1.3, 0.8, 1.1]}}),
    ];
    let base = [
        [-2.0, -2.0, -2.0],
        [2.0, -2.0, -2.0],
        [2.0, 2.0, -2.0],
        [-2.0, 2.0, -2.0],
        [-2.0, -2.0, 2.0],
        [2.0, -2.0, 2.0],
        [2.0, 2.0, 2.0],
        [-2.0, 2.0, 2.0],
    ];
    let asymmetric = [
        [-2.2, -1.8, -2.0],
        [2.1, -1.7, -2.1],
        [2.1, 1.9, -2.2],
        [-2.0, 1.7, -1.8],
        [-1.8, -1.9, 2.0],
        [2.1, -1.7, 2.2],
        [2.2, 1.8, 2.3],
        [-2.1, 1.6, 1.7],
    ];
    for (index, (from, to)) in base.iter().zip(asymmetric).enumerate() {
        operations.push(vertex_move(
            "cage",
            index,
            [to[0] - from[0], to[1] - from[1], to[2] - from[2]],
        ));
    }
    let weights = (0..25)
        .map(|index| 0.15 + 0.2 * (index % 5) as f64)
        .collect::<Vec<_>>();
    operations.extend(group_operations(&weights));
    operations.push(json!({"op": "modifier.create", "target": {"id": "subject"},
        "id": "deform", "type": "mesh_deform",
        "params": {"object": "cage", "precision": precision,
            "use_dynamic_bind": dynamic, "vertex_group": "Influence",
            "invert_vertex_group": invert}}));
    operations.push(bind("deform"));
    operations.push(vertex_move("cage", 0, [0.0, 0.0, 0.12]));
    operations.push(vertex_move("cage", 2, [0.1, -0.03, 0.0]));
    operations.push(vertex_move("cage", 6, [-0.04, 0.08, -0.06]));
    Value::Array(operations)
}

#[test]
fn mesh_deform_uniform_cage_translation_moves_every_vertex() -> TestResult {
    let directory = tempdir()?;
    let operations = mesh_operations(None, 3, false, false, cage_translation_operations());
    let (output, scene) = build_scene(directory.path(), "mesh_static", &operations)?;
    applied(&output)?;
    let expected = (0..25)
        .map(|index| {
            let rest = flat_subject(index);
            [
                rest[0] + CAGE_TRANSLATION[0],
                rest[1] + CAGE_TRANSLATION[1],
                rest[2] + CAGE_TRANSLATION[2],
            ]
        })
        .collect::<Vec<_>>();
    assert_close(
        &evaluated_subject(&scene)?,
        &expected,
        MESH_TOLERANCE,
        "mesh deform static translation",
    )
}
#[test]
fn mesh_deform_nonuniform_bind_transform_matches_blender_translation() -> TestResult {
    let directory = tempdir()?;
    let mut operations = mesh_operations(None, 3, false, false, cage_translation_operations());
    operations[0]["transform"] = json!({
        "translation": [0.2, -0.1, 0.5],
        "rotation": [0.0, 0.0, 0.149_438_132_473_599_22, 0.988_771_077_936_042_2],
        "scale": [0.9, 1.2, 0.75],
    });
    operations[1]["params"]["size"] = json!(6.0);
    operations[1]["transform"] = json!({
        "translation": [-0.4, 0.3, -0.2],
        "rotation": [0.0, 0.0, -0.099_833_416_646_828_15, 0.995_004_165_278_025_8],
        "scale": [1.3, 0.8, 1.1],
    });
    let (output, scene) = build_scene(directory.path(), "mesh_transformed", &operations)?;
    applied(&output)?;
    let expected = (0..25)
        .map(|index| {
            let rest = flat_subject(index);
            [
                // A uniform cage-local translation T maps to inv(S_lin) * C_lin * T in subject-local space
                // (S_lin: subject world linear, C_lin: cage world linear); the harmonic weights reproduce
                // translations exactly, so every vertex is rest + that constant.
                rest[0] + 0.295_054_569_956,
                rest[1] - 0.272_824_308_298,
                rest[2] + 0.733_333_333_333,
            ]
        })
        .collect::<Vec<_>>();
    assert_close(
        &evaluated_subject(&scene)?,
        &expected,
        MESH_TOLERANCE,
        "mesh deform nonuniform bind transform",
    )
}
#[test]
fn mesh_deform_precision_dynamic_weights_and_nonuniform_transforms_match_blender() -> TestResult {
    let cases = [
        (
            "mesh_asymmetric_static_p2",
            2,
            false,
            false,
            [
                [-1.0, -1.0, 0.0],
                [-0.5, -0.5, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [1.0, 1.0, 0.0],
            ],
        ),
        (
            "mesh_asymmetric_static_p3",
            3,
            false,
            false,
            [
                [-1.0, -1.0, 0.003_918_329],
                [-0.5, -0.5, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [1.0, 1.0, 0.0],
            ],
        ),
        (
            "mesh_asymmetric_dynamic_p3_invert",
            3,
            true,
            true,
            [
                [-1.0, -1.0, 0.022_203_874],
                [-0.5, -0.5, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [1.0, 1.0, 0.0],
            ],
        ),
    ];
    let directory = tempdir()?;
    let sample_indices = [0, 6, 12, 18, 24];
    for (name, precision, dynamic, invert, expected) in cases {
        let operations = asymmetric_mesh_operations(precision, dynamic, invert);
        let (output, scene) = build_scene(directory.path(), name, &operations)?;
        applied(&output)?;
        let positions = evaluated_subject(&scene)?;
        let samples = sample_indices
            .iter()
            .map(|index| {
                positions
                    .get(*index)
                    .copied()
                    .ok_or_else(|| -> Box<dyn Error> {
                        Box::new(std::io::Error::other(format!(
                            "mesh deform sample vertex {index} is missing"
                        )))
                    })
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        assert_close(&samples, &expected, 1.0e-3, name)?;
    }
    Ok(())
}

#[test]
fn mesh_deform_inverted_group_scales_the_cage_motion() -> TestResult {
    let directory = tempdir()?;
    let weights = (0..25)
        .map(|index| 0.15 + 0.2 * (index % 5) as f64)
        .collect::<Vec<_>>();
    let operations = mesh_operations(
        Some(weights.clone()),
        4,
        false,
        true,
        cage_translation_operations(),
    );
    let (output, scene) = build_scene(directory.path(), "mesh_invert", &operations)?;
    applied(&output)?;
    let expected = (0..25)
        .map(|index| {
            let rest = flat_subject(index);
            let scale = 1.0 - weights[index];
            [
                rest[0] + CAGE_TRANSLATION[0] * scale,
                rest[1] + CAGE_TRANSLATION[1] * scale,
                rest[2] + CAGE_TRANSLATION[2] * scale,
            ]
        })
        .collect::<Vec<_>>();
    assert_close(
        &evaluated_subject(&scene)?,
        &expected,
        MESH_TOLERANCE,
        "mesh deform inverted group",
    )
}

#[test]
fn mesh_deform_dynamic_bind_scales_the_cage_motion_by_group_weight() -> TestResult {
    let directory = tempdir()?;
    let weights = (0..25)
        .map(|index| 0.15 + 0.2 * (index % 5) as f64)
        .collect::<Vec<_>>();
    let operations = mesh_operations(
        Some(weights.clone()),
        2,
        true,
        false,
        cage_translation_operations(),
    );
    let (output, scene) = build_scene(directory.path(), "mesh_dynamic", &operations)?;
    applied(&output)?;
    let expected = (0..25)
        .map(|index| {
            let rest = flat_subject(index);
            let scale = weights[index];
            [
                rest[0] + CAGE_TRANSLATION[0] * scale,
                rest[1] + CAGE_TRANSLATION[1] * scale,
                rest[2] + CAGE_TRANSLATION[2] * scale,
            ]
        })
        .collect::<Vec<_>>();
    assert_close(
        &evaluated_subject(&scene)?,
        &expected,
        MESH_TOLERANCE,
        "mesh deform dynamic group",
    )
}
#[test]
fn mesh_deform_missing_vertex_group_acts_as_no_group() -> TestResult {
    let directory = tempdir()?;
    let mut operations = mesh_operations(None, 3, false, false, cage_translation_operations());
    operations[2]["params"]["vertex_group"] = json!("Missing");
    let (output, scene) = build_scene(directory.path(), "missing_mesh_group", &operations)?;
    applied(&output)?;
    let expected = (0..25)
        .map(|index| {
            let rest = flat_subject(index);
            [
                rest[0] + CAGE_TRANSLATION[0],
                rest[1] + CAGE_TRANSLATION[1],
                rest[2] + CAGE_TRANSLATION[2],
            ]
        })
        .collect::<Vec<_>>();
    assert_close(
        &evaluated_subject(&scene)?,
        &expected,
        MESH_TOLERANCE,
        "mesh deform missing vertex group",
    )
}

#[test]
fn mesh_deform_ignores_cage_and_owner_object_motion_after_bind() -> TestResult {
    let rest = (0..25).map(flat_subject).collect::<Vec<_>>();
    let moves = [
        (
            "static cage object",
            false,
            vec![json!({"op": "node.update", "target": {"id": "cage"},
                "set": {"transform": {"translation": [0.6, -0.3, 0.4],
                    "rotation": [0.0, 0.0, 0.247_403_959_254_522_9, 0.968_912_421_710_644_7]}}})],
        ),
        (
            "dynamic cage object",
            true,
            vec![json!({"op": "node.update", "target": {"id": "cage"},
                "set": {"transform": {"translation": [0.6, -0.3, 0.4],
                    "rotation": [0.0, 0.0, 0.247_403_959_254_522_9, 0.968_912_421_710_644_7]}}})],
        ),
        (
            "owner object",
            false,
            vec![json!({"op": "node.update", "target": {"id": "subject"},
                "set": {"transform": {"translation": [0.5, 0.0, 0.0]}}})],
        ),
    ];
    let directory = tempdir()?;
    for (label, dynamic, after_bind) in moves {
        let operations = mesh_operations(None, 3, dynamic, false, after_bind);
        let name = label.replace(' ', "_");
        let (output, scene) = build_scene(directory.path(), &name, &operations)?;
        applied(&output)?;
        let actual = evaluated_subject(&scene)?;
        assert_close(&actual, &rest, MESH_TOLERANCE, label)?;
    }
    Ok(())
}

fn laplacian_operations(
    group_indices: &[usize],
    iterations: u64,
    invert: bool,
    moved: &[usize],
) -> Value {
    let weights = (0..25)
        .map(|index| {
            if group_indices.contains(&index) {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let mut operations = vec![json!({"op": "node.create", "id": "subject", "kind": "grid",
        "params": {"size": 2.0, "x_subdivisions": 4, "y_subdivisions": 4}})];
    operations.extend(group_operations(&weights));
    operations.push(json!({"op": "modifier.create", "target": {"id": "subject"}, "id": "laplace",
        "type": "laplacian_deform",
        "params": {"vertex_group": "Influence", "iterations": iterations, "invert_vertex_group": invert}}));
    operations.push(bind("laplace"));
    for &index in moved {
        operations.push(vertex_move(
            "subject",
            index,
            [
                0.05 * (index % 3) as f64,
                -0.04 * (index % 2) as f64,
                0.3 + 0.1 * (index % 4) as f64,
            ],
        ));
    }
    Value::Array(operations)
}

fn border_indices() -> Vec<usize> {
    (0..25)
        .filter(|index| index % 5 == 0 || index % 5 == 4 || *index < 5 || *index >= 20)
        .collect()
}

fn interior_indices() -> Vec<usize> {
    (0..25)
        .filter(|index| !border_indices().contains(index))
        .collect()
}

#[test]
fn laplacian_deform_border_anchors_with_three_iterations_match_blender() -> TestResult {
    let directory = tempdir()?;
    let border = border_indices();
    let operations = laplacian_operations(&border, 3, false, &border);
    let (output, scene) = build_scene(directory.path(), "laplacian_iterations", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_laplacian_subject(&scene)?,
        &LAPLACIAN_BORDER_ITERATIONS_3,
        LAPLACIAN_TOLERANCE,
        "laplacian border anchors, 3 iterations",
    )
}

#[test]
fn laplacian_deform_single_iteration_matches_blender() -> TestResult {
    let directory = tempdir()?;
    let border = border_indices();
    let operations = laplacian_operations(&border, 1, false, &border);
    let (output, scene) = build_scene(directory.path(), "laplacian_single", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_laplacian_subject(&scene)?,
        &LAPLACIAN_BORDER_ITERATIONS_1,
        LAPLACIAN_TOLERANCE,
        "laplacian border anchors, 1 iteration",
    )
}

#[test]
fn laplacian_deform_inverted_group_anchors_interior_vertices() -> TestResult {
    let directory = tempdir()?;
    let border = border_indices();
    let interior = interior_indices();
    let operations = laplacian_operations(&border, 1, true, &interior);
    let (output, scene) = build_scene(directory.path(), "laplacian_invert", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_laplacian_subject(&scene)?,
        &LAPLACIAN_INTERIOR_INVERT,
        LAPLACIAN_TOLERANCE,
        "laplacian inverted group",
    )
}

#[test]
fn corrective_smooth_bind_stores_rest_coordinates() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("corrective_bind");
    init(&scene)?;
    let operations = json!([
        {"op": "node.create", "id": "subject", "kind": "grid",
            "params": {"size": 2.0, "x_subdivisions": 4, "y_subdivisions": 4}},
        {"op": "modifier.create", "target": {"id": "subject"}, "id": "smooth",
            "type": "corrective_smooth", "params": {}},
        {"op": "modifier.bind", "target": {"id": "subject"}, "modifier_id": "smooth"},
    ]);
    applied(&apply(&scene, &operations)?)?;
    let doc = load_doc(&scene)?;
    let node = doc
        .nodes
        .get(&Id::new("subject")?)
        .ok_or("subject node is missing")?;
    let binding = node.modifiers[0]
        .binding_data
        .clone()
        .ok_or("corrective smooth bind data is missing")?;
    if binding["format"] != "blender_corrective_bind_v1" {
        return Err(format!("unexpected corrective bind format: {}", binding["format"]).into());
    }
    let coords = binding["coords"]
        .as_array()
        .ok_or("corrective bind coords are missing")?;
    if coords.len() != 25
        || coords[0] != json!([-1.0, -1.0, 0.0])
        || coords[24] != json!([1.0, 1.0, 0.0])
    {
        return Err(format!("unexpected corrective rest coordinates: {coords:?}").into());
    }
    Ok(())
}

#[test]
fn corrective_smooth_unbind_clears_bind_data() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("corrective_unbind");
    init(&scene)?;
    let operations = json!([
        {"op": "node.create", "id": "subject", "kind": "grid",
            "params": {"size": 2.0, "x_subdivisions": 4, "y_subdivisions": 4}},
        {"op": "modifier.create", "target": {"id": "subject"}, "id": "smooth",
            "type": "corrective_smooth", "params": {}},
        {"op": "modifier.bind", "target": {"id": "subject"}, "modifier_id": "smooth"},
        {"op": "modifier.unbind", "target": {"id": "subject"}, "modifier_id": "smooth"},
    ]);
    applied(&apply(&scene, &operations)?)?;
    let doc = load_doc(&scene)?;
    let node = doc
        .nodes
        .get(&Id::new("subject")?)
        .ok_or("subject node is missing")?;
    if node.modifiers[0].binding_data.is_some() {
        return Err("unbind must clear corrective smooth bind data".into());
    }
    Ok(())
}
