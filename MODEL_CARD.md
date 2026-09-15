# Model card: vEMDestripe

## Intended use

vEMDestripe is intended for research correction of section-to-section intensity variation in registered volume electron microscopy stacks. It accepts grayscale XY sections and an independently screened, same-domain reference pool. Outputs should be assessed in XY, XZ, and YZ views before quantitative biological analysis.

## Out-of-scope use

The model is not validated for clinical diagnosis, autonomous biological conclusions, general photographic enhancement, arbitrary stripe orientation, registration correction, missing-section reconstruction, or replacement of raw-data calibration. It does not prove that a restored structure is biologically real.

## Inputs and outputs

- input section: grayscale, normalized internally from uint8 to `[0,1]`;
- reference: floating-point mean of screened sections followed by 16×16 block averaging for the selected checkpoint;
- output: one restored uint8 section;
- native training size: 1024×1024;
- larger/different sizes: overlapping tiled inference.

## Limitations

Performance can change with microscope, specimen, staining, voxel anisotropy, registration quality, bit depth, artifact severity, reference selection, and the availability of normal sections. The packaged pipeline converts inputs to uint8, so it does not preserve native 16-bit dynamic range. A contaminated or anatomically unrepresentative reference can bias brightness and contrast. Full-reference simulation results do not establish performance on unlabeled real volumes.

## Evaluation and reporting

Report raw outputs, the exact reference filenames, number of references, block size, test Z range, checkpoint checksum, and whether ground truth was available during reference construction. Treat global-affine matched outputs as a sensitivity analysis rather than the primary model result.

## Training data

The model is trained on paired simulations generated from clean vEM volumes using the supplied oblique-boundary simulator. Source data are not redistributed here; users must verify the original licenses and construct independent train/test splits.

