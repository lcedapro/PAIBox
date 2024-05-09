import argparse
from pathlib import Path

import numpy as np
from model import fcnet_2layer_dual_port

import paibox as pb

from data.MNIST_data.ubyte2npy import load_data

def getNetParam():
    param_dict = {}
    timestep = 4
    layer_num = 2
    delay = layer_num - 1

    weights_dir = Path("./weights")
    w1 = np.load(weights_dir / "fc1_weight_np.npy").astype(np.int8).T
    vthr1 = int(np.load(weights_dir / "Vthr1.npy") / 1.0)
    w2 = np.load(weights_dir / "fc2_weight_np.npy").astype(np.int8).T
    vthr2 = int(np.load(weights_dir / "Vthr2.npy") / 1.0)

    param_dict["timestep"] = timestep
    param_dict["layer_num"] = layer_num
    param_dict["delay"] = delay
    param_dict["w1"] = w1
    param_dict["vthr1"] = vthr1
    param_dict["w2"] = w2
    param_dict["vthr2"] = vthr2

    return param_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", help="visualize the input data", action="store_true"
    )
    args = parser.parse_args()

    param_dict = getNetParam()
    pb_net = fcnet_2layer_dual_port(
        param_dict["w1"],
        param_dict["vthr1"],
        param_dict["w2"],
        param_dict["vthr2"],
    )

    # Load_data
    train_images, train_labels = load_data("./data/MNIST_data/train-images.idx3-ubyte", "./data/MNIST_data/train-labels.idx1-ubyte", 60000)
    test_images, test_labels = load_data("./data/MNIST_data/t10k-images.idx3-ubyte", "./data/MNIST_data/t10k-labels.idx1-ubyte", 10000)
    train_num_correct = 0
    test_num_correct = 0
    for i in range(70000):
        # Progress bar
        if i % 100 == 0:
            print("\rProgress: " + str(i) + "/70000", end="")
        
        # Network simulation
        if i < 60000:
            raw_data = train_images[i,:,:]
        else:
            raw_data = test_images[i - 60000,:,:]
        input_data = raw_data.flatten()

        # Visualize
        if args.verbose:
            pe = pb.simulator.PoissonEncoder()
            data_to_see = pe(raw_data).astype(np.int8)
            print(data_to_see)

        # Input
        pb_net.i1.input = input_data[:392]
        pb_net.i2.input = input_data[392:]

        # Simulation, duration=timestep + delay
        sim = pb.Simulator(pb_net)
        sim.run(param_dict["timestep"] + param_dict["delay"], reset=False)

        # Decode the output
        spike_out1 = sim.data[pb_net.probe1].astype(np.int8)
        spike_out2 = sim.data[pb_net.probe2].astype(np.int8)
        spike_out = np.concatenate((spike_out1, spike_out2), axis=1)
        spike_sum = spike_out.sum(axis=0)

        pred = np.argmax(spike_sum)

        # assert pred == 7, print("failed")  # Correct result is 7

        if i < 60000:
            if pred == train_labels[i][0]:
                train_num_correct += 1
        else:
            if pred == test_labels[i - 60000][0]:
                test_num_correct += 1
        # else:
        #     print("failed")
    print("\n")
    print("Accuracy in train dataset: ", train_num_correct / 60000)
    print("Accuracy in test dataset: ", test_num_correct / 10000)

    out_dir = Path(__file__).parent

    mapper = pb.Mapper()
    mapper.build(pb_net)
    graph_info = mapper.compile(
        weight_bit_optimization=True, grouping_optim_target="both"
    )
    mapper.export(
        write_to_file=True,
        fp=out_dir / "debug",
        format="npy",
        split_by_coordinate=True,
        local_chip_addr=(0, 0),
        export_core_params=False,
    )

    # Clear all the results
    mapper.clear()
