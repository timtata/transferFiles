# copy files from source to here or point them here
echo "Setting up CMSSW....................."
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc700
cmsrel CMSSW_12_0_0_pre3
cd CMSSW_12_0_0_pre3/src
cmsenv
scram b clean
scram b -j4
echo "Done CMSSW............................."
cd $currDir
python run_reco_energy.py
