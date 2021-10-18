[![doi](https://img.shields.io/badge/doi-10.3174/ajnr.A5858-brightgreen.svg)](https://doi.org/10.3174/ajnr.A5858)

# TopologyRadiomics Slicer Extension

Topology descriptors are designed to capture subtle sharpness and curvature differences along the surface of diseased pathologies on imaging.

These are based on the hypothesis that local structural changes through infiltration and disruption of disease in a region is likely to cause shape irregularities and in turn, resulting in changes along the surface of the lesion.

# Table of Contents
- [Slicer](#slicer)
  - [Overview](#overview)
- [Tutorial](#tutorial)
- [Contact](#contact)
- [References](#references)


# Slicer
_[Back to **Table of Contents**](#table-of-contents)_

The TopologyRadiomics Slicer 3D extension allows a user to run the `topology_radiomics` algorithm on 3D images and then visualize and save the results. This is done by providing an input surface via segmentation.

## Overview
_[Back to **Table of Contents**](#table-of-contents)_

The general operational flow for using the Slicer extension is to load a 3D image, segment a relavent portion, configure the parameters, and run the algorithm.

Once this is completed, the user can take the output model(s) and visualize them within Slicer. Slicer also supports exporting in various formats, e.g. `.vtk`, which can then be loaded by the Python package `pyvista`.

# Tutorial
_[Back to **Table of Contents**](#table-of-contents)_

Here is a complete demonstration of loading sample data into Slicer, segmenting a tumor, visualizing the output, and saving it to a file.

[![Collage Demonstration](Tutorials/TopologyRadiomicsFullDemo.png?raw=true)](https://youtu.be/-we5zZ7MVRU "Topology Radiomics Full Demo")

To load the surface in Python and visualize it, use the code below:

```python
import pyvista as pv, plotly.graph_objects as go
surface = pv.read('/Users/test/Documents/Segmentation_Segment_1.vtk')

faces = surface.faces.reshape((surface.n_faces, 4))
assert (faces[:, 0] == 3).all()
plotly_3d_meshes = [
  go.Mesh3d(x=surface.points[:, 0], y=surface.points[:, 1], z=surface.points[:, 2],
            i=faces[:, 1], j=faces[:, 2], k=faces[:, 3],
            intensity=surface.point_data[measure], showlegend=True, name=measure)
  for measure in ['curvedness', 'shape_index', 'sharpness', 'total_curvature']
]
fig = go.Figure(data=plotly_3d_meshes,
                layout={'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1}})
fig.show()
```


# Contact
_[Back to **Table of Contents**](#table-of-contents)_

Please report any issues or feature requests via the [Issue Tracker](https://github.com/radxtools/topology-radiomics/issues).

Additional information can be found on the [BrIC Lab](http://bric-lab.com) website.



# References
_[Back to **Table of Contents**](#table-of-contents)_

[![doi](https://img.shields.io/badge/doi-10.3174/ajnr.A5858-brightgreen.svg)](https://doi.org/10.3174/ajnr.A5858)

<a href="http://bric-lab.com"><img align="right" height=100 src="https://static.wixstatic.com/media/a0e8e5_809a649f13254ff293405c7476004e20~mv2.png/v1/fill/w_248,h_240,al_c,usm_0.66_1.00_0.01/a0e8e5_809a649f13254ff293405c7476004e20~mv2.png"></a>

If you make use of this implementation, please cite the following paper:

Ismail, M., Hill, V., Statsevych, V., Huang, R., Prasanna, P., Correa, R., Singh, G., Bera, K., Beig, N., Thawani, R. Madabhushi, A., Aahluwalia, M, and Tiwari, P., "Shape features of the lesion habitat to differentiate brain tumor progression from pseudoprogression on routine multiparametric MRI: a multisite study". American Journal of Neuroradiology, 2018, 39(12), pp.2187-2193.

