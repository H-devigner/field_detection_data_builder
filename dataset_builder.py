import argparse
import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from skimage.transform import resize


def load_geojson(geojson_path):
    """Load GeoJSON file using geopandas."""
    gdf = gpd.read_file(geojson_path)
    return gdf


def clip_raster_to_geometry(raster_path, geometry, crop_to_geometry=True):
    """Clip the raster to the given geometry."""
    with rasterio.open(raster_path) as src:
        geom = [mapping(geometry)]
        out_image, out_transform = mask(src, geom, crop=crop_to_geometry, nodata=0)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )
    return out_image, out_meta


def resize_image(image, target_size):
    """Resize the image to the target size."""
    # Assuming image is (bands, height, width)
    resized = resize(image, (image.shape[0], *target_size), preserve_range=True)
    return resized.astype(image.dtype)


def select_bands(image, num_bands):
    """Select the first num_bands bands."""
    return image[:num_bands]


def create_mask(geometry, shape, transform):
    """Create a binary mask by rasterizing the geometry."""
    from rasterio import features

    mask = features.rasterize(
        [mapping(geometry)],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    return mask


def build_dataset(geojson_path, tiff_dir, output_dir, target_size, num_bands):
    """Build the dataset from GeoJSON and TIFF files."""
    gdf = load_geojson(geojson_path)
    os.makedirs(output_dir, exist_ok=True)

    import glob

    tiff_files = glob.glob(os.path.join(tiff_dir, "**/*.[tj][pi][2f]"), recursive=True)

    # Assuming one TIFF per geometry or match by some logic; here, simplistic assumption
    for idx, row in gdf.iterrows():
        geometry = row.geometry
        for tiff_path in tiff_files:
            try:
                image, meta = clip_raster_to_geometry(tiff_path, geometry)

                # Select bands
                image = select_bands(image, num_bands)

                # Resize
                image_resized = resize_image(image, target_size)

                # Create mask
                mask = create_mask(geometry, target_size, meta["transform"])

                # Save
                np.save(os.path.join(output_dir, f"image_{idx}.npy"), image_resized)
                np.save(os.path.join(output_dir, f"mask_{idx}.npy"), mask)

                print(f"Processed geometry {idx} with {tiff_path}")
                break
            except ValueError as e:
                if "do not overlap" in str(e):
                    continue
                else:
                    raise
        else:
            print(f"No overlapping raster found for geometry {idx}")


def main():
    parser = argparse.ArgumentParser(
        description="Framework to build dataset from Sentinel-2 TIFF and GeoJSON."
    )
    parser.add_argument("--geojson", required=True, help="Path to GeoJSON file")
    parser.add_argument(
        "--tiff_dir", required=True, help="Directory containing Sentinel-2 TIFF files"
    )
    parser.add_argument(
        "--output_dir", default="dataset", help="Output directory for dataset"
    )
    parser.add_argument(
        "--size", default="256,256", help="Target shape size (height,width)"
    )
    parser.add_argument("--bands", type=int, default=3, help="Number of bands to use")

    args = parser.parse_args()

    target_size = tuple(map(int, args.size.split(",")))
    build_dataset(args.geojson, args.tiff_dir, args.output_dir, target_size, args.bands)


if __name__ == "__main__":
    main()
