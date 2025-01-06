# Depth Pro Plugin
A FiftyOne plugin for applying the Apple Depth Pro model to your dataset

### Plugin Overview

## Installation

If you haven't already, install FiftyOne:

```shell
pip install -U fiftyone
```

Then, install the plugin:

```shell
fiftyone plugins download https://github.com/harpreetsahota204/depthpro-plugin
```

Then, install requirements for the plug-in:

```shell
fiftyone plugins requirements @harpreetsahota/depth_pro_plugin --install
```

### Depth Types

The plugin supports two types of depth outputs:

- **Regular Depth**: Direct physical distance measurement in meters. Preferred for autonomous driving, precise measurements, and safety-critical applications. Better for accurate obstacle detection and motion planning.

- Linear depth representation
- Better for absolute distance measurements
- Creating 3D reconstructions
- Common in autonomous driving

- **Inverse Depth**: Reciprocal of depth (1/distance). Better for visualizing near-field details, indoor environments, and SLAM applications. Provides more detail in close ranges where depth changes are significant.

- Better visualization of nearby objects
- More detail in close range
- Doing visual SLAM or Structure from Motion
- Visualizing indoor environments


## Usage in FiftyOne App

You can compute the depth map directly through the FiftyOne App:

1. Launch the FiftyOne App with your dataset
2. Open the "Operators Browser" by clicking on the Operator Browser icon above the sample grid or by typing backtick (`)
3. Type "depth_pro_estimator"
4. Configure the following parameters:
   - **Depth Type**: Choose between:
     - `inverse` - Reciprocal of depth (1/distance)
     - `regular` - Direct physical distance measurement in meters.
   - **Field Name**: Enter the name for the heatmap field (e.g., "depth_map")
5. Click "Execute" to compute depth estimation for your dataset

## Operators

### `depth_pro_estimator`

Computes

## Operator usage via SDK

Once the plugin has been installed, you can instantiate the operator as follows:

```python
import fiftyone.operators as foo

depthpro = foo.get_operator("@harpreetsahota/depth_pro_plugin/depth_pro_estimator")
```

You can then compute the depth map on your dataset by running the operator with your desired parameters:

```python
# Run the operator on your dataset
depthpro(
    dataset,
    depth_field="depth_map", 
    depth_type="inverse",
    delegate=True
    )
```

If you're running in a notebook, it's recommended to launch a [Delegated operation](https://docs.voxel51.com/plugins/using_plugins.html#delegated-operations) by running `fiftyone delegated launch` in terminal, then run as follows:

```python
await depthpro(
    dataset,
    depth_field="depth_map",
    depth_type="inverse",
    delegate=True
    )
```


# Citation

You can read the paper [here](https://arxiv.org/abs/2410.02073).

```bibtex
@article{Bochkovskii2024:arxiv,
  author     = {Aleksei Bochkovskii and Ama\"{e}l Delaunoy and Hugo Germain and Marcel Santos and
               Yichao Zhou and Stephan R. Richter and Vladlen Koltun}
  title      = {Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
  journal    = {arXiv},
  year       = {2024},
  url        = {https://arxiv.org/abs/2410.02073},
}
```