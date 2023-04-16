use std::error::Error;

use clap::Parser;
use n2dmap::normal_to_depth;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    input_file: String,
    output_file: String,
}
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let normal_map = image::open(&args.input_file)?.into_rgb8();
    let depth_map = normal_to_depth(normal_map)?;
    depth_map.save(&args.output_file)?;
    Ok(())
}
