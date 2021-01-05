from adjustText import adjust_text
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import os
import scipy.stats as scs
from scipy.stats import gaussian_kde
import pandas as pd
import pickle
import urllib

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 2021_Golden_AAMDC.py
# This script accompanies the 2021 Nature Communications manuscript by Golden et al, exploring the dependence of breast
#  cancer cell lines on the gene AAMDC (previously C11orf67), using data from the 'DepMap' project and producing
#  visualisations of RNA-seq data comparing specific drugs against shRNA-mediated AAMDC knockdown.
#
# Golden, E et al. [Blancafort, P.]. (2021). The oncogene AAMDC links PI3K-AKT-mTOR signaling with metabolic
#  reprograming in estrogen receptor-positive breast cancer. Nature Communications. In press.
#       DOI: not-yet-known
#
# For further information on this code please contact jcursons (details below), for details on the scientific work
#  please contact the corresponding author Assoc. Prof. Pilar Blancafort
#   pilar.blancafort (at) uwa (dot) edu (dot) au
#
# The authors would like to convey their appreciation for developers of open source modules/dependencies (listed below)
#  as well as those who have contributed to the DepMap project, including scientists, software developers, and patients
#  who have generously donated tissue samples. Further information on DepMap is given below within the DepMapTools
#  class, although users are encouraged to visit: https://depmap.org/portal/depmap/
#
# Script written by Joe Cursons
#   joe.cursons (at) gmail (dot) com
#   github.com/jcursons
#
# Last modified by J Cursons 4th Jan 2021
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Dependencies:
#   - adjust_text
#   - matplotlib
#   - numpy
#   - pandas
#   - scipy
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class PathDir:
    # A module to handle paths (hopefully OS independent); if users wish to manually download the DepMap data then
    #  these paths can be edited/hard-coded to the appropriate location
    #   #   #   #   #   #

    pathCurrDir = os.getcwd()

    pathDataFolder = os.path.join(pathCurrDir, 'data')
    if not os.path.exists(pathDataFolder):
        os.mkdir(pathDataFolder)

    pathPlotFolder = os.path.join(pathCurrDir, 'fig')
    if not os.path.exists(pathPlotFolder):
        os.mkdir(pathPlotFolder)

class PreProc:

    def download_missing_file(url, filepath):
        # a function to download files from specified paths as required

        if not os.path.exists(filepath):
            print(f'Cannot detect the following file:\n\t\t{filepath}'
                  f'\n\tDownloading via figshare, this may take some time..')
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                with open(filepath, 'wb') as outfile:
                    outfile.write(response.read())
        return ()

    def density_scatters(flagResult=False,
                         arrayXIn=np.zeros(1, dtype=np.float),
                         arrayYIn=np.zeros(1, dtype=np.float)):
        arrayJointDist = np.vstack([arrayXIn, arrayYIn])
        arrayJointProb = gaussian_kde(arrayJointDist)(arrayJointDist)

        arrayIndexByZPos = arrayJointProb.argsort()
        arrayXToPlot, arrayYToPlot, arrayZForColor = \
            arrayXIn[arrayIndexByZPos], \
            arrayYIn[arrayIndexByZPos], \
            arrayJointProb[arrayIndexByZPos]

        return arrayXToPlot, arrayYToPlot, arrayZForColor


class DepMapTools:
    # A module to handle the downloading and processing of Cellular Dependency Map (DepMap) data. Further information
    #  on this project can be found at: https://depmap.org/portal/depmap/
    #
    # Data can be obtained from the DepMap project: https://depmap.org/portal/download/
    #  To aid with running this script, hardcoded links via FigShare have been encoded. If users wish to manually
    #   download these files (perhaps best for security) the appropriate links/filenames may need to be hard-coded.
    #
    # Most of these files are available as csv files although pre-processed files are saved using the pickle format
    #  to improve run times when re-running this script.
    #   #   #   #   #   #

    def dict_broadid_to_cclename(flagResult=False):
        # a function to process the metadata table and create a dictionary mapping CCLE/DepMap IDs
        dfMeta = DepMapTools.cell_line_metadata()

        arrayNeitherIDNull = np.bitwise_and(dfMeta['DepMap_ID'].notnull().values.astype(np.bool),
                                            dfMeta['CCLE_Name'].notnull().values.astype(np.bool))

        dictBroadToCCLE = dict(zip(dfMeta['DepMap_ID'].iloc[np.where(arrayNeitherIDNull)[0]].values.tolist(),
                                   dfMeta['CCLE_Name'].iloc[np.where(arrayNeitherIDNull)[0]].values.tolist()))

        return dictBroadToCCLE

    def cell_line_metadata(strMetaDataFilename='sample_info.csv'):

        urlFigShare = "https://ndownloader.figshare.com/files/22629137"
        pathFile = os.path.join(PathDir.pathDataFolder, strMetaDataFilename)

        # if necessary download the file
        PreProc.download_missing_file(urlFigShare, pathFile)

        # read the CSV into a dataframe using pandas
        dfMetaData = pd.read_csv(pathFile,
                                 sep=',',
                                 header=0,
                                 index_col=None)

        return dfMetaData

    def all_demeter2_scores(strD2ScoreFilename='D2_combined_gene_dep_scores.csv',
                            strTempFileNameBase='D2_all'):

        strTempFileName = strTempFileNameBase + '.pickle'

        if os.path.exists(os.path.join(PathDir.pathDataFolder, strTempFileName)):
            print('Loading pre-processed DEMETER2 (siRNA) data..')
            dfD2Scores = pd.read_pickle(os.path.join(PathDir.pathDataFolder, strTempFileName))

        else:
            print('Pre-processing DEMETER2 (siRNA) data..')

            urlFigShare = "https://ndownloader.figshare.com/files/13515395"
            pathFile = os.path.join(PathDir.pathDataFolder, strD2ScoreFilename)
            PreProc.download_missing_file(urlFigShare, pathFile)

            # read the CSV into a dataframe using pandas
            dfIn = pd.read_csv(pathFile,
                               sep=',', header=0, index_col=0)
            listIndex = dfIn.index.tolist()

            # clean up the gene names and retain HGNC symbols for easier indexing
            listHGNCOnly = [strIndex.split(' (')[0] for strIndex in listIndex]
            dfIn['HGNC'] = pd.Series(data=listHGNCOnly, index=dfIn.index.tolist())
            dfD2Scores = dfIn.set_index('HGNC', drop=True)

            # save a temporary file for faster re-processing
            dfD2Scores.to_pickle(os.path.join(PathDir.pathDataFolder, strTempFileName))

        return dfD2Scores

    def all_ceres_scores(strCScoreFilename='Achilles_gene_effect.csv',
                         strTempFileNameBase='CERES_all',
                         flagPerformExtraction=False):

        strTempFileName = strTempFileNameBase + '.pickle'

        if not os.path.exists(os.path.join(PathDir.pathDataFolder, strTempFileName)):
            flagPerformExtraction = True

        if flagPerformExtraction:
            print('Pre-processing CERES (CRISPRi) data..')

            urlFigShare = "https://ndownloader.figshare.com/files/22629068"
            pathFile = os.path.join(PathDir.pathDataFolder, strCScoreFilename)
            PreProc.download_missing_file(urlFigShare, pathFile)

            # read the CSV into a dataframe using pandas
            dfIn = pd.read_csv(pathFile,
                               sep=',', header=0, index_col=0)
            listColumns = dfIn.columns.tolist()

            # clean up the gene names and retain HGNC symbols for easier indexing
            listHGNCOnly = [strColumn.split(' (')[0] for strColumn in listColumns]
            dfIn.rename(columns=dict(zip(listColumns, listHGNCOnly)), inplace=True)

            # transpose the dataframe
            dfCScores = dfIn.transpose()

            # save a temporary file for faster re-processing
            dfCScores.to_pickle(os.path.join(PathDir.pathDataFolder, strTempFileName))
        else:
            print('Loading pre-processed CERES (CRISPRi) data..')
            dfCScores = pd.read_pickle(os.path.join(PathDir.pathDataFolder, strTempFileName))

        return dfCScores

    def all_rnaseq_data(strRNASeqFilename='CCLE_expression.csv',
                        strTempFileName='DepMap_RNAseq.pickle',
                        flagPerformExtraction=False):

        if not os.path.exists(os.path.join(PathDir.pathDataFolder, strTempFileName)):
            flagPerformExtraction = True

        if flagPerformExtraction:
            print('Pre-processing DepMap RNA-seq data..')

            # load the dictionary for mapping Broad/DepMap IDs to CCLE cell line names
            dictBroadIDToCCLE = DepMapTools.dict_broadid_to_cclename()

            urlFigShare = "https://ndownloader.figshare.com/files/22897979"
            strFilename = os.path.join(PathDir.pathDataFolder, strRNASeqFilename)
            PreProc.download_missing_file(urlFigShare, strFilename)

            # this CSV is relatively big and contains log-transformed transcript abundance data; to reduce the
            #  memory footprint, first read the first row to obtain column names
            dfRNASeqHeader = pd.read_csv(
                os.path.join(PathDir.pathDataFolder, strRNASeqFilename),
                sep=',', nrows=0, header=0, index_col=0)
            listColumns = dfRNASeqHeader.columns.tolist()
            # use the column names to create a dictionary specifying data type as float16 (values are well within the
            #  appropriate abundance range for a 16-bit float given log transformation)
            dictDataTypes = dict(zip(listColumns, [np.float16] * len(listColumns)))

            # read the CSV into a dataframe using pandas, specifying the data type
            dfRNASeq = pd.read_csv(os.path.join(PathDir.pathDataFolder, strRNASeqFilename), sep=',',
                               header=0, index_col=0, dtype=dictDataTypes)
            listIndex = dfRNASeq.index.tolist()

            # ensure there are no missing dictionary values before using this to rename the cell lines
            for strLine in set(listIndex).difference(set(dictBroadIDToCCLE.keys())):
                dictBroadIDToCCLE[strLine] = f'failed_map:{strLine}'
            listRowRename = [dictBroadIDToCCLE[strRow] for strRow in listIndex]
            dfRNASeq.rename(index=dict(zip(listIndex, listRowRename)), inplace=True)

            # save a temporary file for faster re-processing
            dfRNASeq.to_pickle(os.path.join(PathDir.pathDataFolder, strTempFileName))

        else:
            dfRNASeq = pd.read_pickle(os.path.join(PathDir.pathDataFolder, strTempFileName))

        return dfRNASeq


    def copynumber_data(strCNVDataFile='CCLE_gene_cn.csv',
                        flagPerformExtraction=False):

        urlFigShare = "https://ndownloader.figshare.com/files/22629107"
        strFilename = os.path.join(PathDir.pathDataFolder, strCNVDataFile)
        PreProc.download_missing_file(urlFigShare, strFilename)

        strTempFile = strFilename.replace('.csv', '.pickle')

        if not os.path.exists(os.path.join(PathDir.pathDataFolder, strTempFile)):
            flagPerformExtraction = True

        if flagPerformExtraction:
            dfCNVHeader = pd.read_csv(os.path.join(PathDir.pathDataFolder, strCNVDataFile),
                                         sep='\t', nrows=0, header=0, index_col=0)
            listColumns = dfCNVHeader.columns.tolist()

            dictDataTypes = dict(zip(listColumns, [np.float16] * len(listColumns)))

            dfCNV = pd.read_csv(os.path.join(PathDir.pathDataFolder, strCNVDataFile),
                                  header=0, sep=',', index_col=0, dtype=dictDataTypes)

            dfCNV.to_pickle(os.path.join(PathDir.pathDataFolder, strTempFile))

        else:

            dfCNV = pd.read_pickle(os.path.join(PathDir.pathDataFolder, strTempFile))

        return dfCNV

class ENSEMBLTools:

    def grc_human_mapping(flagResult=False,
                          numRelease=98):

        if numRelease > 75:
            strDataFile = f'Homo_sapiens.GRCh38.{numRelease}.gtf.gz'
        elif numRelease <= 75:
            strDataFile = f'Homo_sapiens.GRCh37.{numRelease}.gtf.gz'
        # note that this needs to be fixed for anything pre GRCh37 but I don't know why you would want to use that

        pathFile = os.path.join(PathDir.pathDataFolder, strDataFile)

        if numRelease == 98:
            # hard-code a link for release 98 to make running the script easier
            urlENSEMBL = 'http://ftp.ensembl.org/pub/release-98/gtf/homo_sapiens/Homo_sapiens.GRCh38.98.chr.gtf.gz'
            PreProc.download_missing_file(urlENSEMBL, pathFile)

        dfMapping = pd.read_csv(pathFile,
                                sep='\t',
                                compression='gzip',
                                header=None,
                                comment='#')

        return dfMapping

    def dict_gtf_ensg_to_hgnc(flagResult=False,
                              numRelease=98,
                              flagPerformExtraction=False):

        strTempFilename = f'GRCh38_{numRelease}_ENSGToHGNC.pickle'

        if not os.path.exists(os.path.join(PathDir.pathDataFolder, strTempFilename)):
            flagPerformExtraction=True

        if flagPerformExtraction:

            dfEnsDB = ENSEMBLTools.grc_human_mapping(numRelease=numRelease)

            if numRelease >= 75:
                arrayGeneRowIndices = np.where((dfEnsDB.iloc[:,2]=='gene').values.astype(np.bool))[0]
            else:
                arrayGeneRowIndices = np.where((dfEnsDB.iloc[:,2]=='exon').values.astype(np.bool))[0]
            numGenes = len(arrayGeneRowIndices)

            listGenes = [None]*numGenes
            listGeneENSG = [None]*numGenes

            strFirstGeneDetails = dfEnsDB.iloc[arrayGeneRowIndices[0],8]
            listFirstGeneDetails = strFirstGeneDetails.split(';')
            numGeneNameIndex = np.where(['gene_name "' in strDetails for strDetails in listFirstGeneDetails])[0][0]
            numGeneIDIndex = np.where(['gene_id "' in strDetails for strDetails in listFirstGeneDetails])[0][0]

            for iGene in range(numGenes):
                strGeneDetails = dfEnsDB.iloc[arrayGeneRowIndices[iGene],8]
                listGeneDetails = strGeneDetails.split(';')

                strGene = listGeneDetails[numGeneNameIndex].split('gene_name "')[1].strip('"')
                strENSG = listGeneDetails[numGeneIDIndex].split('gene_id "')[1].strip('"')

                listGenes[iGene] = strGene
                listGeneENSG[iGene] = strENSG

            if len(listGeneENSG) > len(set(listGeneENSG)):

                dfMapped = pd.DataFrame({'ENSG':listGeneENSG, 'HGNC':listGenes})
                dfMapped.drop_duplicates(subset='ENSG', inplace=True)

                dictEnsGeneToHGNC = dict(zip(dfMapped['ENSG'].values.tolist(),
                                             dfMapped['HGNC'].values.tolist()))
            else:
                dictEnsGeneToHGNC = dict(zip(listGeneENSG, listGenes))

            with open(os.path.join(PathDir.pathDataFolder, strTempFilename), 'wb') as handFile:
                pickle.dump(dictEnsGeneToHGNC, handFile, protocol=pickle.HIGHEST_PROTOCOL)
        else:

            with open(os.path.join(PathDir.pathDataFolder, strTempFilename), 'rb') as handFile:
                dictEnsGeneToHGNC = pickle.load(handFile)

        return dictEnsGeneToHGNC

class Load:

    listDiffExprFiles = ['SUM52-AAMDC_EV_vs_WT_-_DESeq2_geneLevel.csv',
                         'SUM52-AAMDC_shRNA_vs_EV_-_DESeq2_geneLevel.csv',
                         'SUM52-AAMDC_Dacto_vs_DMSO_-_DESeq2_geneLevel.csv',
                         'SUM52-AAMDC_Evero_vs_DMSO_-_DESeq2_geneLevel.csv',
                         'SUM52-AAMDC_AZD_vs_DMSO_-_DESeq2_geneLevel.csv',
                         'SUM52-AAMDC_Bupar_vs_DMSO_-_DESeq2_geneLevel.csv']

    listAbundFiles = ['Salmon_tximport_geneLevel-DrugInhib.csv',
                      'Salmon_tximport_geneLevel-shRNA.csv']

    dictDrugExpToSample = {'A2': 'DMSO-2',
                           'A3': 'DMSO-3',
                           'A4': 'DMSO-4',
                           'B1': 'Dactoclisib-1',
                           'B3': 'Dactoclisib-3',
                           'B4': 'Dactoclisib-4',
                           'C2': 'Everolimus-2',
                           'C3': 'Everolimus-3',
                           'C4': 'Everolimus-4',
                           'D1': 'AZD8055-1',
                           'D2': 'AZD8055-2',
                           'D3': 'AZD8055-3',
                           'E1': 'Buparlisib-1',
                           'E2': 'Buparlisib-2',
                           'E4': 'Buparlisib-4'}

    dictshRNAExpToSample = {'Sum52ORFsh2Replicate1': 'shRNA-1',
                            'Sum52ORFsh2Replicate2': 'shRNA-2',
                            'Sum52ORFsh2Replicate3': 'shRNA-3',
                            'Sum52PlkoEVReplicate1': 'sh.EV-1',
                            'Sum52PlkoEVReplicate2': 'sh.EV-2',
                            'Sum52PlkoEVReplicate3': 'sh.EV-3',
                            'Sum52wildtypeReplicate1': 'sh.WT-1',
                            'Sum52wildtypeReplicate2': 'sh.WT-2',
                            'Sum52wildtypeReplicate3': 'sh.WT-3'}

    def rna_diffexpr(flagPerformExtraction=True,
                     listFileNames=listDiffExprFiles):

        dictENSGToHGNC = ENSEMBLTools.dict_gtf_ensg_to_hgnc()

        listDFToMerge = []
        for iDiffExpr in range(len(listFileNames)):
            strFileName = listFileNames[iDiffExpr]
            strExp = strFileName.split('_-_DESeq2_geneLevel.csv')[0]
            strComp = strExp.split('SUM52-AAMDC_')[1]

            dfIn = pd.read_table(os.path.join(PathDir.pathDataFolder, strFileName),
                                 sep=',', header=0, index_col=0)
            listColumns = dfIn.columns.tolist()
            listColRenamed = [strComp + ':' + strCol for strCol in listColumns]
            dfIn.rename(columns=dict(zip(listColumns, listColRenamed)),
                        inplace=True)

            listStatCol = [strCol for strCol in listColRenamed if 'stat' in strCol]
            listBaseMeanCol = [strCol for strCol in listColRenamed if 'baseMean' in strCol]
            listAdjPValCol = [strCol for strCol in listColRenamed if 'padj' in strCol]

            arrayAdjPVal = dfIn[listAdjPValCol].values.astype(np.float)

            arrayIsNullPVal = dfIn[listAdjPValCol].isnull().values.astype(np.bool)

            arrayIsNullStatVal = dfIn[listStatCol].isnull().values.astype(np.bool)

            arrayIsLowBaseMean = dfIn[listBaseMeanCol].values.astype(np.float) < 10

            arrayIsStrongStatVal = np.abs(np.nan_to_num(dfIn[listStatCol].values.astype(np.float))) > 4.0

            arrayNullLowBaseMeanIndices = np.where(np.bitwise_and(arrayIsNullPVal, arrayIsLowBaseMean))[0]
            arrayNullStatIndices = np.where(np.bitwise_and(arrayIsNullPVal, arrayIsNullStatVal))[0]

            arrayIsWTF = np.bitwise_and(np.bitwise_and(arrayIsNullPVal, arrayIsStrongStatVal),
                                        ~arrayIsLowBaseMean)
            print('{}'.format(
                np.sum(arrayIsWTF)) + ' genes with NaN adj. p-value but abs(stat.) value > 4.. setting 1E-3')

            arrayAdjPVal[arrayNullLowBaseMeanIndices] = 1.0
            arrayAdjPVal[arrayNullStatIndices] = 1.0
            arrayAdjPVal[np.where(arrayIsWTF)[0]] = 1E-3

            dfIn[listAdjPValCol[0]] = pd.Series(arrayAdjPVal[:, 0], index=dfIn.index.tolist())

            listDFToMerge.append(dfIn)

        dfRNA = pd.concat(listDFToMerge, axis=1, join='outer')
        listGenesENSG = dfRNA.index.tolist()
        listRNAGenes = []
        for strGene in listGenesENSG:
            strHGNC = 'failed_map'
            if strGene in dictENSGToHGNC.keys():
                strHGNC = dictENSGToHGNC[strGene]
            listRNAGenes.append(strHGNC)

        dfRNA['HGNC'] = pd.Series(listRNAGenes, index=dfRNA.index.tolist())

        return dfRNA

    def rna_abund(flagPerformExtraction=True,
                  listFileNames=listAbundFiles):

        dictENSGToHGNC = ENSEMBLTools.dict_gtf_ensg_to_hgnc()

        listDFToMerge = []
        for iDiffExpr in range(len(listFileNames)):
            strFileName = listFileNames[iDiffExpr]

            dfIn = pd.read_csv(os.path.join(PathDir.pathDataFolder, strFileName),
                                 sep=',', header=0, index_col=0)
            listColumns = dfIn.columns.tolist()
            listAbundColumns = [strCol for strCol in listColumns if 'abundance.' in strCol]

            dfTemp = dfIn[listAbundColumns].copy(deep=True)
            listAbundColumnsClean = [strCol.split('abundance.')[1] for strCol in listAbundColumns]
            dfTemp.rename(columns=dict(zip(listAbundColumns, listAbundColumnsClean)),
                          inplace=True)

            if 'DrugInhib' in strFileName:
                dfTemp.rename(columns=Load.dictDrugExpToSample,
                              inplace=True)
            else:
                dfTemp.rename(columns=Load.dictshRNAExpToSample,
                              inplace=True)

            listDFToMerge.append(dfTemp)

        dfRNA = pd.concat(listDFToMerge, axis=1, join='outer')
        listGenesENSG = dfRNA.index.tolist()
        listRNAGenes = []
        for strGene in listGenesENSG:
            strHGNC = 'failed_map'
            if strGene in dictENSGToHGNC.keys():
                strHGNC = dictENSGToHGNC[strGene]
            listRNAGenes.append(strHGNC)

        dfRNA['HGNC'] = pd.Series(listRNAGenes, index=dfRNA.index.tolist())

        return dfRNA

class Plot:

    numFontSize = 10
    listOutFormats = ['png', 'pdf']

    dictCompToDisp = {'EV_vs_WT':'shRNA EVC $vs.$ shRNA NTC',
                      'shRNA_vs_EV':'shRNA C11orf67 $vs.$ shRNA EVC',
                      'Dacto_vs_DMSO':'Dactoclisib $vs.$ DMSO control',
                      'Evero_vs_DMSO':'Everolimus $vs.$ DMSO control',
                      'AZD_vs_DMSO':'AZD8055 $vs.$ DMSO control',
                      'Bupar_vs_DMSO':'Buparlisib $vs.$ DMSO control'}

    def brca_ceres_assoc(flagResult=False):

        dictSubtypeOutLabel = {'HER2_amp':'HER2$^{amp}$',
                               'basal': 'Basal',
                               'basal_A': 'Basal A',
                               'basal_B': 'Basal B',
                               'luminal': 'Luminal',
                               'luminal_HER2_amp': 'Luminal HER2$^{amp}$'}

        dictBroadToCCLE = DepMapTools.dict_broadid_to_cclename()
        dictCCLEToBroad = dict(zip(dictBroadToCCLE.values(), dictBroadToCCLE.keys()))

        dfCERES = DepMapTools.all_ceres_scores()
        listCERESLines = dfCERES.columns.tolist()
        listCERESLinesCCLE = [dictBroadToCCLE[strLine] for strLine in listCERESLines]

        dfRNA = DepMapTools.all_rnaseq_data()
        listRNALines = dfRNA.index.tolist()
        listRNAGenes = dfRNA.columns.tolist()
        strRNACol = [strGene for strGene in listRNAGenes if 'AAMDC (' in strGene][0]

        dfMeta = DepMapTools.cell_line_metadata()
        arrayBreastRowIndices = \
            np.where(dfMeta['primary_disease']=='Breast Cancer')[0]
        listBreastLines = dfMeta['CCLE_Name'].iloc[arrayBreastRowIndices].values.tolist()
        listSubtype = dfMeta['lineage_molecular_subtype'].iloc[arrayBreastRowIndices].values.tolist()
        for iLine in range(len(listSubtype)):
            if not (listSubtype[iLine] == listSubtype[iLine]):
                listSubtype[iLine] = '-'


        listUniqueSubtype = sorted(list(set(listSubtype)))
        listUniqueSubtype.remove('-')
        listUniqueSubtype.append('-')
        dictSubtypeToLines = dict()
        for strSubtype in listUniqueSubtype:
            listLinesForSubtype = [listBreastLines[i] for i in range(len(listBreastLines))
                                   if listSubtype[i]==strSubtype]
            dictSubtypeToLines[strSubtype] = listLinesForSubtype

        dfCNV = DepMapTools.copynumber_data()
        listCNVGenes = dfCNV.columns.tolist()
        listCNVLines = dfCNV.index.tolist()
        listCNVLinesCCLE = [dictBroadToCCLE[strLine] for strLine in listCNVLines if strLine in dictBroadToCCLE.keys()]
        strCNVCol = [strGene for strGene in listCNVGenes if 'AAMDC (' in strGene][0]

        listSharedCNVCERESLines = list(set(listCERESLinesCCLE).intersection(set(listCNVLinesCCLE)))
        listSharedCNVCERESBreastLines = [strLine for strLine in listSharedCNVCERESLines if '_BREAST' in strLine]
        listSharedCNVCERESBreastLinesBroad = [dictCCLEToBroad[strLine] for strLine in listSharedCNVCERESBreastLines]

        listSharedRNACERESLines = list(set(listCERESLinesCCLE).intersection(set(listRNALines)))
        listSharedRNACERESBreastLines = [strLine for strLine in listSharedRNACERESLines if '_BREAST' in strLine]
        listSharedRNACERESBreastLinesBroad = [dictCCLEToBroad[strLine] for strLine in listSharedRNACERESBreastLines]


        arrayColorNorm = matplotlib.colors.Normalize(vmin=0, vmax=9)

        arrayColorMap = matplotlib.cm.ScalarMappable(norm=arrayColorNorm, cmap=plt.cm.tab10)

        handFig = plt.figure(figsize=(5,4))

        handAx = handFig.add_axes([0.15, 0.15, 0.60, 0.75])

        listToLabel = []
        listToLabelBroad = []
        for iSubtype in range(len(listUniqueSubtype)):
            strSubtype = listUniqueSubtype[iSubtype]
            listLinesForSubtype = dictSubtypeToLines[strSubtype]
            listSubtypeAndDataLines = list(set(listSharedCNVCERESBreastLines).intersection(set(listLinesForSubtype)))
            listSubtypeAndDataLinesBroad = [dictCCLEToBroad[strLine] for strLine in listSubtypeAndDataLines]
            if len(listSubtypeAndDataLinesBroad) > 0:
                plt.scatter(dfCNV[strCNVCol].reindex(listSubtypeAndDataLinesBroad),
                            dfCERES[listSubtypeAndDataLinesBroad].loc['AAMDC'],
                            c=[arrayColorMap.to_rgba(iSubtype)]*len(listSubtypeAndDataLinesBroad),
                            alpha=0.8,
                            s=5,
                            label=dictSubtypeOutLabel[strSubtype])
                listToLabel += listSubtypeAndDataLines
                listToLabelBroad += listSubtypeAndDataLinesBroad

        listHandText = [plt.text(dfCNV[strCNVCol].loc[listToLabelBroad[i]],
                                 dfCERES[listToLabelBroad[i]].loc['AAMDC'],
                                 listToLabel[i].split('_BREAST')[0],
                                 fontsize=Plot.numFontSize * 0.4,
                                 ha='center', zorder=8)
                        for i in range(len(listToLabelBroad))]

        print('Attempting to fit text labels..')
        adjust_text(listHandText,
                    force_text=(0.1, 0.1),
                    expand_text=(1.05, 1.05),
                    force_points=(0.2, 0.2),
                    arrowprops=dict(arrowstyle='-',
                                    color='k', lw=0.5,
                                    connectionstyle="arc3",
                                    alpha=0.9))
        for handText in listHandText:
            handText.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'),
                                       path_effects.Normal()])

        handAx.set_xlabel('Copy Number Variance (chr. seg.)', fontsize=Plot.numFontSize)
        handAx.set_ylabel('DepMap CERES score', fontsize=Plot.numFontSize)

        plt.legend(loc='upper left',
                   bbox_to_anchor=(1.01, 0.9),
                   fontsize=Plot.numFontSize * 0.5,
                   scatterpoints=1,
                   framealpha=1,
                   ncol=1)

        for strFormat in Plot.listOutFormats:
            handFig.savefig(os.path.join(PathDir.pathPlotFolder, f'DepMap_BRCA_CNV_vs_CERES.{strFormat}'),
                            ext=strFormat, dpi=300)
        plt.close(handFig)


        handFig = plt.figure(figsize=(5,4))

        listToLabel = []
        listToLabelBroad = []
        handAx = handFig.add_axes([0.15, 0.15, 0.60, 0.75])
        for iSubtype in range(len(listUniqueSubtype)):
            strSubtype = listUniqueSubtype[iSubtype]
            listLinesForSubtype = dictSubtypeToLines[strSubtype]
            listSubtypeAndDataLines = list(set(listSharedRNACERESBreastLines).intersection(set(listLinesForSubtype)))
            listSubtypeAndDataLinesBroad = [dictCCLEToBroad[strLine] for strLine in listSubtypeAndDataLines]

            plt.scatter(dfRNA[strRNACol].reindex(listSubtypeAndDataLines),
                        dfCERES[listSubtypeAndDataLinesBroad].loc['AAMDC'],
                            c=[arrayColorMap.to_rgba(iSubtype)]*len(listSubtypeAndDataLinesBroad),
                            alpha=0.8,
                            s=5,
                        label=strSubtype)

            listToLabel += listSubtypeAndDataLines
            listToLabelBroad += listSubtypeAndDataLinesBroad

        listHandText = [plt.text(dfRNA[strRNACol].loc[listToLabel[i]],
                                 dfCERES[listToLabelBroad[i]].loc['AAMDC'],
                                 listToLabel[i].split('_BREAST')[0],
                                 fontsize=Plot.numFontSize * 0.4,
                                 ha='center', zorder=8)
                        for i in range(len(listToLabelBroad))]

        print('Attempting to fit text labels..')
        adjust_text(listHandText,
                    force_text=(0.1, 0.1),
                    expand_text=(1.05, 1.05),
                    force_points=(0.2, 0.2),
                    arrowprops=dict(arrowstyle='-',
                                    color='k', lw=0.5,
                                    connectionstyle="arc3",
                                    alpha=0.9))
        for handText in listHandText:
            handText.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'),
                                       path_effects.Normal()])

        handAx.set_xlabel('RNA abundance (log(TPM))', fontsize=Plot.numFontSize)
        handAx.set_ylabel('DepMap CERES score', fontsize=Plot.numFontSize)

        plt.legend(loc='upper left',
                   bbox_to_anchor=(1.01, 0.9),
                   fontsize=Plot.numFontSize * 0.7,
                   scatterpoints=1,
                   framealpha=1,
                   ncol=1)

        for strFormat in Plot.listOutFormats:
            handFig.savefig(os.path.join(PathDir.pathPlotFolder, f'DepMap_BRCA_RNA_vs_CERES.{strFormat}'),
                            ext=strFormat, dpi=300)
        plt.close(handFig)

        return flagResult

    def brca_demeter_assoc(flagResult=False):

        dictBroadToCCLE = DepMapTools.dict_broadid_to_cclename()
        dictCCLEToBroad = dict(zip(dictBroadToCCLE.values(), dictBroadToCCLE.keys()))

        dfDEMETER = DepMapTools.all_demeter2_scores()
        listDEMETERLines = dfDEMETER.columns.tolist()


        dfRNA = DepMapTools.all_rnaseq_data()
        listRNALines = dfRNA.index.tolist()
        listRNAGenes = dfRNA.columns.tolist()
        strRNACol = [strGene for strGene in listRNAGenes if 'AAMDC (' in strGene][0]

        dfMeta = DepMapTools.cell_line_metadata()
        arrayBreastRowIndices = \
            np.where(dfMeta['primary_disease']=='Breast Cancer')[0]
        listBreastLines = dfMeta['CCLE_Name'].iloc[arrayBreastRowIndices].values.tolist()
        listSubtype = dfMeta['lineage_molecular_subtype'].iloc[arrayBreastRowIndices].values.tolist()
        for iLine in range(len(listSubtype)):
            if not (listSubtype[iLine] == listSubtype[iLine]):
                listSubtype[iLine] = '-'

        listBreastLinesInDEMETER = [strLine for strLine in listBreastLines if strLine in listDEMETERLines]

        listUniqueSubtype = sorted(list(set(listSubtype)))
        listUniqueSubtype.remove('-')
        listUniqueSubtype.append('-')
        dictSubtypeToLines = dict()
        for strSubtype in listUniqueSubtype:
            listLinesForSubtype = [listBreastLines[i] for i in range(len(listBreastLines))
                                   if listSubtype[i]==strSubtype]
            dictSubtypeToLines[strSubtype] = listLinesForSubtype

        dfCNV = DepMapTools.copynumber_data()
        listCNVGenes = dfCNV.columns.tolist()
        listCNVLines = dfCNV.index.tolist()
        listCNVLinesCCLE = [dictBroadToCCLE[strLine] for strLine in listCNVLines if strLine in dictBroadToCCLE.keys()]
        strCNVCol = [strGene for strGene in listCNVGenes if 'AAMDC (' in strGene][0]

        listSharedCNVDEMETERLines = list(set(listDEMETERLines).intersection(set(listCNVLinesCCLE)))
        listSharedCNVDEMETERBreastLines = [strLine for strLine in listSharedCNVDEMETERLines if '_BREAST' in strLine]
        listSharedCNVDEMETERBreastLinesBroad = [dictCCLEToBroad[strLine] for strLine in listSharedCNVDEMETERBreastLines]

        listSharedRNADEMETERLines = list(set(listDEMETERLines).intersection(set(listRNALines)))
        listSharedRNADEMETERBreastLines = [strLine for strLine in listSharedRNADEMETERLines if '_BREAST' in strLine]
        listSharedRNADEMETERBreastLinesBroad = [dictCCLEToBroad[strLine] for strLine in listSharedRNADEMETERBreastLines]


        arrayColorNorm = matplotlib.colors.Normalize(vmin=0, vmax=9)

        arrayColorMap = matplotlib.cm.ScalarMappable(norm=arrayColorNorm, cmap=plt.cm.tab10)

        handFig = plt.figure(figsize=(5,4))

        handAx = handFig.add_axes([0.15, 0.15, 0.60, 0.75])

        for iSubtype in range(len(listUniqueSubtype)):
            strSubtype = listUniqueSubtype[iSubtype]
            listLinesForSubtype = dictSubtypeToLines[strSubtype]
            listSubtypeAndDataLines = list(set(listSharedCNVDEMETERBreastLines).intersection(set(listLinesForSubtype)))
            listSubtypeAndDataLinesBroad = [dictCCLEToBroad[strLine] for strLine in listSubtypeAndDataLines]
            plt.scatter(dfCNV[strCNVCol].reindex(listSubtypeAndDataLinesBroad),
                        dfDEMETER[listSubtypeAndDataLines].loc['AAMDC'],
                        c=[arrayColorMap.to_rgba(iSubtype)]*len(listSubtypeAndDataLinesBroad),
                        alpha=0.8,
                        s=5,
                        label=strSubtype)

        # listHandText = [plt.text(dfCNV[strCNVCol].loc[listSharedCNVDEMETERBreastLinesBroad[i]],
        #                          dfDEMETER[listSharedCNVDEMETERBreastLines[i]].loc['AAMDC'],
        #                          listSharedCNVDEMETERBreastLines[i].split('_BREAST')[0],
        #                          fontsize=Plot.numFontSize * 0.4,
        #                          ha='center', zorder=8)
        #                 for i in range(len(listSharedCNVDEMETERBreastLinesBroad))]
        #
        # print('Attempting to fit text labels..')
        # adjust_text(listHandText,
        #             force_text=(0.1, 0.1),
        #             expand_text=(1.05, 1.05),
        #             force_points=(0.1, 0.1),
        #             arrowprops=dict(arrowstyle='-',
        #                             color='k', lw=0.5,
        #                             connectionstyle="arc3",
        #                             alpha=0.9))
        # for handText in listHandText:
        #     handText.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'),
        #                                path_effects.Normal()])

        handAx.set_xlabel('Copy Number Variance (chr. seg.)', fontsize=Plot.numFontSize)
        handAx.set_ylabel('DepMap DEMETER score', fontsize=Plot.numFontSize)

        plt.legend(loc='upper left',
                   bbox_to_anchor=(1.01, 0.9),
                   fontsize=Plot.numFontSize * 0.7,
                   scatterpoints=1,
                   framealpha=1,
                   ncol=1)

        handFig.savefig(os.path.join(PathDir.pathPlotFolder, 'DepMap_BRCA_CNV_vs_DEMETER.png'),
                        ext='png', dpi=300)
        plt.close(handFig)


        handFig = plt.figure(figsize=(5,4))

        handAx = handFig.add_axes([0.15, 0.15, 0.60, 0.75])
        for iSubtype in range(len(listUniqueSubtype)):
            strSubtype = listUniqueSubtype[iSubtype]
            listLinesForSubtype = dictSubtypeToLines[strSubtype]
            listSubtypeAndDataLines = list(set(listSharedRNADEMETERBreastLines).intersection(set(listLinesForSubtype)))
            listSubtypeAndDataLinesBroad = [dictCCLEToBroad[strLine] for strLine in listSubtypeAndDataLines]

            plt.scatter(dfRNA[strRNACol].reindex(listSubtypeAndDataLines),
                        dfDEMETER[listSubtypeAndDataLines].loc['AAMDC'],
                            c=[arrayColorMap.to_rgba(iSubtype)]*len(listSubtypeAndDataLinesBroad),
                            alpha=0.8,
                            s=5,
                        label=strSubtype)
        handAx.set_xlabel('RNA abundance (log(TPM))', fontsize=Plot.numFontSize)
        handAx.set_ylabel('DepMap DEMETER score', fontsize=Plot.numFontSize)


        listHandText = [plt.text(dfRNA[strRNACol].loc[listSharedRNADEMETERBreastLines[i]],
                                 dfDEMETER[listSharedRNADEMETERBreastLines[i]].loc['AAMDC'],
                                 listSharedRNADEMETERBreastLines[i].split('_BREAST')[0],
                                 fontsize=Plot.numFontSize * 0.4,
                                 ha='center', zorder=8)
                        for i in range(len(listSharedRNADEMETERBreastLines))]

        print('Attempting to fit text labels..')
        adjust_text(listHandText,
                    force_text=(0.1, 0.1),
                    expand_text=(1.05, 1.05),
                    force_points=(0.1, 0.1),
                    arrowprops=dict(arrowstyle='-',
                                    color='k', lw=0.5,
                                    connectionstyle="arc3",
                                    alpha=0.9))
        for handText in listHandText:
            handText.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'),
                                       path_effects.Normal()])


        plt.legend(loc='upper left',
                   bbox_to_anchor=(1.01, 0.9),
                   fontsize=Plot.numFontSize * 0.7,
                   scatterpoints=1,
                   framealpha=1,
                   ncol=1)

        handFig.savefig(os.path.join(PathDir.pathPlotFolder, 'DepMap_BRCA_RNA_vs_DEMETER2.png'),
                        ext='png', dpi=300)
        plt.close(handFig)

        return flagResult

    def drug_effects_vs_shrna_effects(flagResult=False):

        listPairsForComp = [['shRNA_vs_EV', 'AZD_vs_DMSO'],
                            ['shRNA_vs_EV', 'Bupar_vs_DMSO'],
                            ['shRNA_vs_EV', 'Dacto_vs_DMSO'],
                            ['shRNA_vs_EV', 'Evero_vs_DMSO'],
                            ['shRNA_vs_EV', 'EV_vs_WT']]

        dfAbund = Load.rna_abund()
        listAbundColumns = dfAbund.columns.tolist()
        listAbundDataColumns = listAbundColumns.copy()
        listAbundDataColumns.remove('HGNC')

        listHGNCSymbol = dfAbund['HGNC'].values.tolist()
        listHGNCSymbolClean = []
        for strGene in listHGNCSymbol:
            if strGene == 'failed_map':
                listHGNCSymbolClean.append('')
            else:
                listHGNCSymbolClean.append(strGene)

        dfAbund['HGNC'] = pd.Series(listHGNCSymbolClean, index=dfAbund.index.tolist())

        dfAbund[['HGNC']+listAbundDataColumns].to_csv(
            os.path.join(PathDir.pathDataFolder, 'TPM_Abund.tsv'), sep='\t')

        arrayAboveAbundThresh = \
            np.sum(np.nan_to_num(dfAbund[listAbundDataColumns].values.astype(np.float)) > 1, axis=1) >= 3

        listGeneAboveAbundThresh = dfAbund.iloc[np.where(arrayAboveAbundThresh)[0],:].index.tolist()

        dfDiffExprIn = Load.rna_diffexpr()

        dfDiffExpr = dfDiffExprIn.reindex(listGeneAboveAbundThresh)

        listCol = dfDiffExpr.columns.tolist()

        listFCCol = [strCol for strCol in listCol if 'log2FoldChange' in strCol]
        listConds = [strCol.split(':log2FoldChange')[0] for strCol in listFCCol]

        listOutCols = ['HGNC']
        for strCond in listConds:
            listOutCols.append(strCond + ':log2FoldChange')
            listOutCols.append(strCond + ':padj')

        for iComp in range(len(listPairsForComp)):
            strCondOne = listPairsForComp[iComp][0]
            strCondTwo = listPairsForComp[iComp][1]

            arrayXData = np.nan_to_num(dfDiffExpr[strCondOne + ':log2FoldChange'].values.astype(np.float))
            arrayYData = np.nan_to_num(dfDiffExpr[strCondTwo + ':log2FoldChange'].values.astype(np.float))

            arrayCondOneSig = np.bitwise_and(np.nan_to_num(dfDiffExpr[strCondOne + ':padj'].values.astype(np.float)) < 0.01,
                                             dfDiffExpr[strCondOne + ':padj'].notnull())

            arrayCondTwoSig = np.bitwise_and(np.nan_to_num(dfDiffExpr[strCondTwo + ':padj'].values.astype(np.float)) < 0.01,
                                             dfDiffExpr[strCondTwo + ':padj'].notnull())

            arrayIsSigInEither = np.bitwise_or(arrayCondOneSig, arrayCondTwoSig)
            arraySigInEitherIndices = np.where(arrayIsSigInEither)[0]


            arrayGridSpec = matplotlib.gridspec.GridSpec(
                nrows=1, ncols=2,
                left=0.10, right=0.98,
                bottom=0.15, top=0.94,
                wspace=0.4)

            handFig = plt.figure()
            handFig.set_size_inches(w=8, h=4)

            handAx = plt.subplot(arrayGridSpec[0])

            slope, intercept, r_value, p_value, std_err = scs.linregress(arrayXData,
                                                                         arrayYData)
            structSpearCorr = scs.spearmanr(arrayXData,
                                            arrayYData)

            arrayXToPlot, arrayYToPlot, arrayColor = \
                PreProc.density_scatters(arrayXIn=arrayXData, arrayYIn=arrayYData)

            handAx.scatter(arrayXToPlot,
                           arrayYToPlot,
                           c=arrayColor,
                           cmap=plt.cm.magma,
                           s=8,
                           linewidth=0.0,
                           alpha=0.6)
            arrayXValsForLine = [-9.0, 9.0]
            arrayYValsForLine = [slope*i + intercept for i in arrayXValsForLine]
            handAx.plot(arrayXValsForLine, arrayYValsForLine, color='g', linestyle='--', linewidth=1.0, alpha=0.5)

            handAx.text(x=-7.6, y=7.7,
                        s='$y=$' + '{:03.2f}'.format(slope) + '$*x + $' + '{:03.2f}'.format(intercept) +
                          '\n$R^{2}=$' + '{:03.2f}'.format(r_value ** 2) +
                          '; $n_{genes}=$' + '{}'.format(len(arrayXData)) + '\n' +
                          '$r_{S}$ = ' + '{:03.2f}'.format(structSpearCorr[0]),
                        fontsize=Plot.numFontSize * 0.8,
                        ha='left', va='top',
                        path_effects=[path_effects.withStroke(linewidth=2, foreground="w")]
                        )

            handAx.set_xlabel(Plot.dictCompToDisp[strCondOne] + '\nlog$_{2}$FC')
            handAx.set_ylabel(Plot.dictCompToDisp[strCondTwo] + '\nlog$_{2}$FC')
            handAx.set_title('All genes')
            handAx.set_xlim([-8, 8])
            handAx.set_ylim([-8, 8])
            handAx.set_xticks([-8, -4, 0, 4, 8])
            handAx.set_yticks([-8, -4, 0, 4, 8])

            handAx = plt.subplot(arrayGridSpec[1])

            arrayXToPlot, arrayYToPlot, arrayColor = \
                PreProc.density_scatters(arrayXIn=arrayXData[arraySigInEitherIndices],
                                         arrayYIn=arrayYData[arraySigInEitherIndices])

            slope, intercept, r_value, p_value, std_err = scs.linregress(arrayXData[arraySigInEitherIndices],
                                                                         arrayYData[arraySigInEitherIndices])
            structSpearCorr = scs.spearmanr(arrayXData[arraySigInEitherIndices],
                                         arrayYData[arraySigInEitherIndices])

            handAx.scatter(arrayXToPlot,
                           arrayYToPlot,
                           c=arrayColor,
                           cmap=plt.cm.magma,
                           s=8,
                           linewidth=0.0,
                           alpha=0.6)
            arrayXValsForLine = [-9.0, 9.0]
            arrayYValsForLine = [slope*i + intercept for i in arrayXValsForLine]
            handAx.plot(arrayXValsForLine, arrayYValsForLine, color='g', linestyle='--', linewidth=1.0, alpha=0.5)

            handAx.text(x=-7.6, y=7.7,
                        s='$y=$'+'{:03.2f}'.format(slope)+'$*x + $'+'{:03.2f}'.format(intercept) +
                          '\n$R^{2}=$' + '{:03.2f}'.format(r_value**2) +
                          '; $n_{genes}=$' + '{}'.format(len(arraySigInEitherIndices)) + '\n' +
                          '$r_{S}$=' + '{:03.2f}'.format(structSpearCorr[0]),
                        fontsize=Plot.numFontSize*0.8,
                        ha='left', va='top',
                        path_effects=[path_effects.withStroke(linewidth=2, foreground="w")]
                        )

            handAx.set_xlabel(Plot.dictCompToDisp[strCondOne] + '\nlog$_{2}$FC')
            handAx.set_ylabel(Plot.dictCompToDisp[strCondTwo] + '\nlog$_{2}$FC')
            handAx.set_title('adjusted $p$-val < 0.01')
            handAx.set_xlim([-8, 8])
            handAx.set_ylim([-8, 8])
            handAx.set_xticks([-8, -4, 0, 4, 8])
            handAx.set_yticks([-8, -4, 0, 4, 8])

            for strFormat in Plot.listOutFormats:
                handFig.savefig(os.path.join(PathDir.pathPlotFolder, f'{strCondOne}_vs_{strCondTwo}.{strFormat}'),
                                ext=strFormat, dpi=300)
            plt.close(handFig)

        return flagResult


_ = Plot.brca_ceres_assoc()

_ = Plot.drug_effects_vs_shrna_effects()