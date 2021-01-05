# Golden_2021_NatComm
A repository to accompany the manuscript:

Golden, E *et al.* [Blancafort, P.]. (2021). The oncogene AAMDC links PI3K-AKT-mTOR signaling with metabolic
reprograming in estrogen receptor-positive breast cancer. *Nature Communications*. In press.

**DOI**: not-yet-known

# Overview
This script accompanies the 2021 Nature Communications manuscript by Golden et al, exploring the dependence of breast
cancer cell lines on the gene AAMDC (previously C11orf67), using data from the 'DepMap' project and producing
visualisations of RNA-seq data comparing specific drugs against shRNA-mediated AAMDC knockdown.

For further information on this code please contact jcursons (repository owner), for details on the scientific work
please contact the corresponding author Assoc. Prof. Pilar Blancafort:
- pilar.blancafort (at) uwa (dot) edu (dot) au
- pilar.blancafort (at) gmail (dot) com

# Data
## The Cellular Dependency Map Project (DepMap)
The authors would like to thank those who have contributed to the DepMap project, including scientists, software
developers, and patients who have generously donated tissue samples. Further information on DepMap is given below
within the DepMapTools class, although users are encouraged to visit: 
- https://depmap.org/portal/depmap/

Functions within the attached script will automatically download these data, although users may wish to
download these data themselves from a security perspective. If so, please ensure that paths within the PathDir
class are consistent with the location of the data, and filenames in the DepMap class are also appropriate.

## Data from Golden *et al.*
A small number of files from the RNA-seq analysis of shRNA and drug treated cells are included within the
data folder to reproduce figure 5b within the associated manuscript.

# Non-standard dependencies
The authors would like to convey their appreciation for developers of open source modules/dependencies, as their
work helped to make this analysis possible.
- [adjustText - automatic label placement for matplotlib](https://github.com/Phlya/adjustText)
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scipy](https://www.scipy.org/)

# Execution
Users should be able to clone the repository and in the presence of the required dependencies (see above)
run this script within their preferred development environment (e.g. PyCharm) or from the command line.

The PathDir module performs some directory checks and creates directories if required (the data directory
should be present from the cloned repository - this is required to reproduce Fig. 5b). DepMap data should 
be automatically downloaded, although please note some of these files are relatively large (700MB) so this
may take some time for users without high speed broadband internet.

Please feel free to contact me (jcursons) if you encounter any issues. 