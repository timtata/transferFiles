filesToTransfer="worker.sh,run_reco_energy.py"
Executable="worker.sh"
jdl_file="condor_runningFile_job.jdl"
log_prefix="condor_runningFile_job"
echo "universe = vanilla">$jdl_file
echo "Executable = worker.sh">>$jdl_file
echo "Should_Transfer_Files = YES">>$jdl_file
echo "WhenToTransferOutput = ON_EXIT_OR_EVICT">>$jdl_file
echo "Transfer_Input_Files = ${filesToTransfer}">>$jdl_file
echo "Output = ${log_prefix}.stdout">>$jdl_file
echo "Error = ${log_prefix}.stderr">>$jdl_file
echo "Log = ${log_prefix}.condor">>$jdl_file
echo "notification = never">>$jdl_file
#echo "Arguments = run_reco_energy.py">>$jdl_file
echo "Queue">>$jdl_file
condor_submit $jdl_file
