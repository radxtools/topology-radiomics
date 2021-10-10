cmake_minimum_required(VERSION 3.13.4)

project(TopologyRadiomics)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/radxtools/topology-radiomics/tree/master/slicer/topologyradiomics")
set(EXTENSION_CATEGORY "Informatics")
set(EXTENSION_CONTRIBUTORS "Pallavi Tiwari (BriC), Satish Viswanath (BriC), Rob Toth (Toth Technology), Nathan Hillyer (Toth Technology), Tom Primozic")
set(EXTENSION_DESCRIPTION "The TopologyRadiomics extension provides a 3D Slicer interface to the topology_radiomics library. topology_radiomics is an open-source Python package for capturing subtle sharpness and curvature differences along the surface of diseased pathologies on imaging.")
set(EXTENSION_ICONURL "https://github.com/radxtools/topology-radiomics/raw/master/slicer/topologyradiomics/TopologyRadiomicsSlicer/Resources/Icons/TopologyRadiomicsSlicer.png")
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