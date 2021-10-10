cmake_minimum_required(VERSION 3.13.4)

project(TopologyRadiomics)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/radxtools/topology-radiomics/tree/master/slicer/topologyradiomics")
set(EXTENSION_CATEGORY "Informatics")
set(EXTENSION_CONTRIBUTORS "Unknown Unknown (Unknown)")
set(EXTENSION_DESCRIPTION "The TopologyRadiomics is being developed.")
set(EXTENSION_ICONURL "TopologyRadiomicsSlicer/Resources/Icons/TopologyRadiomicsSlicer.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/radxtools/topology-radiomics/raw/master/slicer/topologyradiomics/TopologyRadiomicsSlicer/Screenshot.jpg")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(TopologyRadiomicsSlicer)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})