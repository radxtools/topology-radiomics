import math
from typing import Union

import vtk
import slicer
import numpy as np
import pyvista as pv
from slicer.util import VTKObservationMixin
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget, ScriptedLoadableModuleLogic, ScriptedLoadableModuleTest

# try:
#   import topology_radiomics
# except ImportError:
#   slicer.util.pip_install('topology_radiomics')

import sys
sys.path.insert(0, '/Users/tom/my/code/slicer/topology-radiomics/src/')

from topology_radiomics.config import MarchingCubesAlgorithm
from topology_radiomics import MorphologyConfig, compute_morphology_features, convert_volume_into_mask


#
# TopologyRadiomicsSlicer
#

class TopologyRadiomicsSlicer(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "TopologyRadiomics"
    self.parent.categories = ["Informatics"]
    self.parent.dependencies = []
    self.parent.contributors = ["BriC Lab (Case Western University)"]
    self.parent.helpText = """
TopologyRadiomics captures subtle sharpness and curvature differences along the surface of diseased pathologies on imaging.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
If you make use of this implementation, please cite the following paper:

Ismail, M., Hill, V., Statsevych, V., Huang, R., Prasanna, P., Correa, R., Singh, G., Bera, K., Beig, N., Thawani, R. Madabhushi, A., Aahluwalia, M, and Tiwari, P., "Shape features of the lesion habitat to differentiate brain tumor progression from pseudoprogression on routine multiparametric MRI: a multisite study". American Journal of Neuroradiology, 2018, 39(12), pp.2187-2193.
"""

#
# TopologyRadiomicsSlicerWidget
#


class TopologyRadiomicsSlicerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)
    self.ui = None
    self.logic = None
    self.parameterNode = None
    # self.developerMode = True

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    uiWidget = slicer.util.loadUI(self.resourcePath('UI/TopologyRadiomicsSlicer.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)
    uiWidget.setMRMLScene(slicer.mrmlScene)
    self.layout.addStretch(1)

    self.ui.iterSlider.connect('valueChanged(double)', self.updateParameterNodeFromGui)
    self.ui.sigmaSlider.connect('valueChanged(double)', self.updateParameterNodeFromGui)

    self.ui.spacingXSlider.connect('valueChanged(double)', self.updateParameterNodeFromGui)
    self.ui.spacingYSlider.connect('valueChanged(double)', self.updateParameterNodeFromGui)
    self.ui.spacingZSlider.connect('valueChanged(double)', self.updateParameterNodeFromGui)
    self.ui.algorithmComboBox.connect('currentIndexChanged(int)', self.updateParameterNodeFromGui)
    self.ui.stepSizeSlider.connect('valueChanged(double)', self.updateParameterNodeFromGui)
    self.ui.clipSlider.connect('valueChanged(double)', self.updateParameterNodeFromGui)

    self.ui.inputSelector.setMRMLScene(slicer.mrmlScene)
    self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGui)
    self.ui.parameterNodeSelector.addAttribute("vtkMRMLScriptedModuleNode", "ModuleName", self.moduleName)
    self.ui.parameterNodeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.setParameterNode)

    self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

    self.logic = TopologyRadiomicsSlicerLogic()
    self.setParameterNode(self.logic.getParameterNode())

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def setParameterNode(self, inputParameterNode):
    """
    Adds observers to the selected parameter node. Observation is needed because when the
    parameter node is changed then the GUI must be updated immediately.
    """
    if inputParameterNode == self.parameterNode:
      return
    
    wasBlocked = self.ui.parameterNodeSelector.blockSignals(True)
    self.ui.parameterNodeSelector.setCurrentNode(inputParameterNode)
    self.ui.parameterNodeSelector.blockSignals(wasBlocked)

    if self.parameterNode is not None:
      self.removeObserver(self.parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGuiFromParameterNode)
    if inputParameterNode is not None:
      self.addObserver(inputParameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGuiFromParameterNode)
    self.parameterNode = inputParameterNode

    self.updateGuiFromParameterNode()

  def updateGuiState(self):
    if self.parameterNode is None:
      return
    if self.parameterNode.GetNodeReference("Input"):
      self.ui.applyButton.text = "Run"
      self.ui.applyButton.enabled = True
    else:
      self.ui.applyButton.text = "Select input nodes"
      self.ui.applyButton.enabled = False

  def updateGuiFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """
    if self.parameterNode is None:
      return

    wasBlocked = self.ui.inputSelector.blockSignals(True)
    self.ui.inputSelector.setCurrentNode(self.parameterNode.GetNodeReference("Input"))
    self.ui.inputSelector.blockSignals(wasBlocked)

    def get_int(parameter, default):
      try:
        return int(self.parameterNode.GetParameter(parameter))
      except ValueError:
        return default

    def get_float(parameter, default):
      try:
        return float(self.parameterNode.GetParameter(parameter))
      except ValueError:
        return default

    default_config = MorphologyConfig()

    wasBlocked = self.ui.sigmaSlider.blockSignals(True)
    self.ui.sigmaSlider.setValue(get_float('GaussianSigma', default_config.gaussian_sigma))
    self.ui.sigmaSlider.blockSignals(wasBlocked)

    wasBlocked = self.ui.iterSlider.blockSignals(True)
    self.ui.iterSlider.setValue(get_int('GaussianIterations', default_config.gaussian_iterations))
    self.ui.iterSlider.blockSignals(wasBlocked)

    wasBlocked = self.ui.spacingXSlider.blockSignals(True)
    self.ui.spacingXSlider.setValue(get_float('VoxelSpacingX', default_config.voxel_spacing[0]))
    self.ui.spacingXSlider.blockSignals(wasBlocked)

    wasBlocked = self.ui.spacingYSlider.blockSignals(True)
    self.ui.spacingYSlider.setValue(get_float('VoxelSpacingY', default_config.voxel_spacing[1]))
    self.ui.spacingYSlider.blockSignals(wasBlocked)

    wasBlocked = self.ui.spacingZSlider.blockSignals(True)
    self.ui.spacingZSlider.setValue(get_float('VoxelSpacingZ', default_config.voxel_spacing[2]))
    self.ui.spacingZSlider.blockSignals(wasBlocked)

    wasBlocked = self.ui.algorithmComboBox.blockSignals(True)
    self.ui.algorithmComboBox.setCurrentIndex(self.ui.algorithmComboBox.findText(self.parameterNode.GetParameter("MarchingCubesAlgorithm") or default_config.marching_cubes_algorithm.name))
    self.ui.algorithmComboBox.blockSignals(wasBlocked)

    wasBlocked = self.ui.stepSizeSlider.blockSignals(True)
    self.ui.stepSizeSlider.setValue(get_int('MarchingCubesStepSize', default_config.marching_cubes_step_size))
    self.ui.stepSizeSlider.blockSignals(wasBlocked)

    wasBlocked = self.ui.clipSlider.blockSignals(True)
    self.ui.clipSlider.setValue(get_float('ClipPercent', default_config.clip_percent))
    self.ui.clipSlider.blockSignals(wasBlocked)

    self.updateGuiState()

  def updateParameterNodeFromGui(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """
    if self.parameterNode is None:
      return

    with slicer.util.NodeModify(self.parameterNode):
      self.parameterNode.SetNodeReferenceID("Input", self.ui.inputSelector.currentNodeID)
      self.parameterNode.SetParameter('GaussianSigma', str(self.ui.sigmaSlider.value))
      self.parameterNode.SetParameter('GaussianIterations', str(int(self.ui.iterSlider.value)))
      self.parameterNode.SetParameter('VoxelSpacingX', str(self.ui.spacingXSlider.value))
      self.parameterNode.SetParameter('VoxelSpacingY', str(self.ui.spacingYSlider.value))
      self.parameterNode.SetParameter('VoxelSpacingZ', str(self.ui.spacingZSlider.value))
      self.parameterNode.SetParameter('MarchingCubesAlgorithm', self.ui.algorithmComboBox.currentText)
      self.parameterNode.SetParameter('MarchingCubesStepSize', str(int(self.ui.stepSizeSlider.value)))
      self.parameterNode.SetParameter('ClipPercent', str(self.ui.clipSlider.value))

    self.updateGuiState()

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """

    config = MorphologyConfig()
    config.gaussian_sigma = self.ui.sigmaSlider.value
    config.gaussian_iterations = int(self.ui.iterSlider.value)
    config.voxel_spacing = (self.ui.spacingXSlider.value, self.ui.spacingYSlider.value, self.ui.spacingZSlider.value)
    config.marching_cubes_algorithm = getattr(MarchingCubesAlgorithm, self.ui.algorithmComboBox.currentText, None) or config.marching_cubes_algorithm
    config.marching_cubes_step_size = int(self.ui.stepSizeSlider.value)
    config.clip_percent = self.ui.clipSlider.value

    self.ui.applyButton.enabled = False
    self.ui.applyButton.text = 'Working...'
    slicer.app.processEvents()      # make sure the GUI updates
    try:
      self.logic.run(self.ui.inputSelector.currentNode(), config)
    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: %s" % e)
      import traceback
      traceback.print_exc()
    finally:
      self.ui.applyButton.enabled = True
      self.ui.applyButton.text = 'Run'


#
# TopologyRadiomicsSlicerLogic
#

class TopologyRadiomicsSlicerLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def createParameterNode(self):
    """ Override base class method to provide default parameters. """
    node = ScriptedLoadableModuleLogic.createParameterNode(self)
    default_config = MorphologyConfig()
    node.SetParameter('GaussianSigma', str(default_config.gaussian_sigma))
    node.SetParameter('GaussianIterations', str(default_config.gaussian_iterations))
    node.SetParameter('VoxelSpacingX', str(default_config.voxel_spacing[0]))
    node.SetParameter('VoxelSpacingY', str(default_config.voxel_spacing[1]))
    node.SetParameter('VoxelSpacingZ', str(default_config.voxel_spacing[2]))
    node.SetParameter('MarchingCubesAlgorithm', default_config.marching_cubes_algorithm.name)
    node.SetParameter('MarchingCubesStepSize', str(default_config.marching_cubes_step_size))
    node.SetParameter('ClipPercent', str(default_config.clip_percent))

    return node

  @staticmethod
  def get_segment_mask(segmentation_node, segment_name=None, segment_id=None):
    assert (segment_name is None) != (segment_id is None), 'exactly one of `segment_name` and `segment_id` must be provided'
    if segment_name is not None:
      segment_id = segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_name)
    segment_ids = vtk.vtkStringArray()
    segment_ids.InsertNextValue(segment_id)
    labelmap_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    # use EXTENT_UNION_OF_SEGMENTS instead of EXTENT_REFERENCE_GEOMETRY to get much smaller
    # mask - will speed up later processing; origin will reflect that!
    slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsToLabelmapNode(
      segmentation_node, segment_ids, labelmap_volume_node, None, slicer.vtkSegmentation.EXTENT_UNION_OF_SEGMENTS,
    )

    origin = labelmap_volume_node.GetOrigin()
    spacing = labelmap_volume_node.GetSpacing()
    directions = np.zeros((3, 3))
    labelmap_volume_node.GetIJKToRASDirections(directions)
    directions = np.diag(directions)
    transform_matrix = np.zeros((4, 4))
    transform_matrix[:3, :3] = np.diag(spacing * directions)
    transform_matrix[:3, 3] = origin
    transform_matrix[3, 3] = 1

    ijk_to_ras = vtk.vtkMatrix4x4()
    labelmap_volume_node.GetIJKToRASMatrix(ijk_to_ras)
    matrix = slicer.util.arrayFromVTKMatrix(ijk_to_ras)
    assert np.array_equiv(matrix, transform_matrix), 'unexpected transformation matrix:\n%s' % matrix

    image = labelmap_volume_node.GetImageData()
    shape = tuple(reversed(image.GetDimensions()))
    mask = vtk.util.numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(shape)

    slicer.mrmlScene.RemoveNode(labelmap_volume_node)
    return mask, origin, spacing, directions

  @staticmethod
  def model_from_mask(mask, origin, spacing, directions, name=None, config=None, padding=None):
    assert np.array_equiv(np.unique(mask), [0, 1]), 'mask can only contain values 0 (empty space) and 1 (full volume)'

    if config is None:
      config = MorphologyConfig()

    origin, spacing = np.asarray(origin), np.asarray(spacing)

    if padding is None:
      # without padding, our surface might hit the border of the bounding box / array after smoothing
      padding = math.ceil(config.gaussian_sigma * config.gaussian_iterations)
    if padding > 0:
      padded = np.zeros([mask.shape[0] + 2 * padding, mask.shape[1] + 2 * padding, mask.shape[2] + 2 * padding])
      padded[padding:-padding, padding:-padding, padding:-padding] = mask
      mask = padded
      origin -= spacing * directions * padding
    mask = convert_volume_into_mask(mask, merge_labels=[1])

    result = compute_morphology_features(mask, config)
    # recreate PolyData output, after applying the transformation to the vertices (otherwise its position won't
    # match the original model/segment!)
    faces = result.isosurface.faces
    verts = result.isosurface.verts
    verts = verts[:, [2, 1, 0]] * (spacing * directions) / np.array(list(reversed(config.voxel_spacing))) + origin
    poly_faces = np.column_stack([3 * np.ones((faces.shape[0], 1), dtype=np.int), faces])
    polydata = pv.PolyData(verts, poly_faces.flatten())
    model_node = slicer.modules.models.logic().AddModel(polydata)
    if name is not None:
      model_node.SetName(name)

    # add different surface measures, which color the surface; they can be selected in the `Scalars` section
    # of the `Models` module
    MEASURES = ['shape_index', 'curvedness', 'sharpness', 'total_curvature']
    for measure in MEASURES:
      vtk_array = vtk.util.numpy_support.numpy_to_vtk(getattr(result.surface_measures, measure))
      vtk_array.SetName(measure)
      model_node.GetPolyData().GetPointData().AddArray(vtk_array)

    display_node = model_node.GetDisplayNode()
    display_node.SetActiveScalarName(MEASURES[0])
    display_node.AutoScalarRangeOn()
    display_node.SetAndObserveColorNodeID('vtkMRMLColorTableNodeFilePlasma.txt')
    display_node.SetScalarVisibility(True)

    return result

  @classmethod
  def run(cls, input_node: Union[slicer.vtkMRMLSegmentationNode, slicer.vtkMRMLModelNode], config=None):
    if config is None:
      config = MorphologyConfig()

    if isinstance(input_node, slicer.vtkMRMLSegmentationNode):
      segmentation = input_node.GetSegmentation()
      n = segmentation.GetNumberOfSegments()
      for i in range(n):
        segment_id = segmentation.GetNthSegmentID(i)
        segment_name = segmentation.GetSegment(segment_id).GetName()

        mask, origin, spacing, directions = cls.get_segment_mask(input_node, segment_id=segment_id)
        cls.model_from_mask(mask, origin, spacing, directions, name='%s_%s' % (input_node.GetName(), segment_name), config=config)
    elif isinstance(input_node, slicer.vtkMRMLModelNode):
      segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
      slicer.modules.segmentations.logic().ImportModelToSegmentationNode(input_node, segmentation_node)
      assert segmentation_node.GetSegmentation().GetNumberOfSegments()
      segment_id = segmentation_node.GetSegmentation().GetNthSegmentID(0)
      mask, origin, spacing, directions = cls.get_segment_mask(segmentation_node, segment_id=segment_id)
      slicer.mrmlScene.RemoveNode(segmentation_node)
      cls.model_from_mask(mask, origin, spacing, directions, name='%s_topology' % input_node.GetName(), config=config)
    else:
      raise TypeError('expected vtkMRMLSegmentationNode or vtkMRMLModelNode, got %s' % type(input_node))




# TopologyRadiomicsSlicerTest
#

class TopologyRadiomicsSlicerTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
