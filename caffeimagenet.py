from cappuccino.experimentutil import hpolib_experiment_main
from cappuccino.paramutil import hpolib_to_caffenet
from cappuccino.caffeconvnet import CaffeConvNet
from cappuccino.caffeconvnet import TerminationCriterionMaxEpoch, TerminationCriterionTestAccuracy, TerminationCriterionExternalInBackground
from cappuccino.caffeconvnet import TerminationCriterionDivergenceDetection
from cappuccino.caffeconvnet import ImagenetConvNet
import os
import sys
import time
import HPOlib.benchmark_util as benchmark_util

MEAN_PERFORMANCE_ON_LAST = 10

def get_run_num():
    if not os.path.exists("num_run"):
        open("num_run", "w").write("0")
	return 0
    else:
	num = int(open("num_run").read())
        num += 1
        open("num_run", "w").write(str(num))
	return num

def construct_caffeconvnet(params):
    print "CaffeConvNet params:"
    device = "GPU"
    device_id = 0

    print "Device: ", device
    print "Device id: ", device_id
 
    #TODO: move snapshot when network was fully trained!
    caffe = ImagenetConvNet(
			params=params,
			train_file="/misc/lmbraid10/dosovits/Caffe/Data/ILSVRC2012/100classes_500samplesperclass/train/lmdb",
			num_train=50000,
			valid_file="/misc/lmbraid10/dosovits/Caffe/Data/ILSVRC2012/100classes_500samplesperclass/val/lmdb",
		 	mean_file="/misc/lmbraid10/dosovits/Caffe/Data/ILSVRC2012/100classes_500samplesperclass/train/mean.binaryproto",
                        termination_criterions = [TerminationCriterionTestAccuracy(50),
                                                   TerminationCriterionMaxEpoch(200),
                                                   TerminationCriterionDivergenceDetection(),
                                                   TerminationCriterionExternalInBackground(
                                                      external_cmd="python -m pylrpredictor.terminationcriterion --nthreads 3",
                                                      run_every_x_epochs=30)],
                        test_every_x_epoch=0.5,
			num_valid=5000,
			snapshot_prefix="caffnet-run%d" % get_run_num(),
			batch_size_valid=100,
			snapshot_on_exit=1,
			device="GPU",
			device_id=0)
    return caffe


def main(params, **kwargs):
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    working_dir = os.getcwd()
    return hpolib_experiment_main(params, construct_caffeconvnet,
        experiment_dir=experiment_dir,
        working_dir=working_dir,
        mean_performance_on_last=MEAN_PERFORMANCE_ON_LAST)

if __name__ == "__main__":
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
                    ("SAT", abs(duration), result, -1, str(__file__))
