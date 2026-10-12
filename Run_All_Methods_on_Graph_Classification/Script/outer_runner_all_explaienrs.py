import os
import sys
import pickle
import torch
from importlib import reload

from charset_normalizer import from_path

from Datasets_for_Explainability_Methods.Load_Dataset_Script import load_datasets as Load_Dataset_Module
from Meta_Explainer_on_Benchmarking.Train_Models_on_Graph_Classification.Script import create_GNNs as Create_and_Load_GNNS_Module
from Meta_Explainer_on_Benchmarking.Run_All_Methods_on_Graph_Classification.Utils import running_file_checker as Runner_File_Checker


py_path = '/data/cs.aau.dk/ey33jw/Meta_Explainer_on_Benchmarking/'
os.chdir(py_path)
print("Current Working Directory:", os.getcwd())



running_file_checker = Runner_File_Checker.file_checker(save_root="/data/cs.aau.dk/ey33jw/Meta_Explainer_on_Benchmarking")
gnn_oprtimized_hyperparameters_dir = ("/data/cs.aau.dk/ey33jw/Meta_Explainer_on_Benchmarking/"
                                      "HyperParameter_Optimization_of_GNNs_on_Graph_Classification/Experimental_Results")

params_path = os.path.join(gnn_oprtimized_hyperparameters_dir, "optimized_gnn_hyperparameters.pkl")

with open(params_path, "rb") as f:
    all_parameters = pickle.load(f)


# RUN_DATASETS = ["MUTAG", "NCI1", "Graph-SST5", "ENZYMES", "PROTEINS", "IsCyclic", ]
RUN_DATASETS = ["Graph-SST5"]
# RUN_GNNS = ["GCN_plus_GAP_Model", "DGCNN_Model", "DIFFPOOL_Model", "GIN_Model"]
RUN_GNNS = ["GCN_plus_GAP_Model",]
VERIFY_GRAPHMASK_REFERENCE = True
# RUN_EXPLAINERS = ["GNNExplainer", "PGExplainer", "GraphMask", "SubGraphX", "CF2", "PGMExplainer", "XGNN",
#                   "GNNInterpreter", ]
# RUN_EXPLAINERS = ["GNNExplainer", "PGExplainer", "GraphMask", "CF2", "PGMExplainer", "XGNN", "GNNInterpreter", ]
RUN_EXPLAINERS = ["GraphMask"]

GNN_NAME_TO_INDEX_dict = {
    "GCN_plus_GAP_Model": 1,
    "DGCNN_Model": 2,
    "DIFFPOOL_Model": 3,
    "GIN_Model": 4,
}

EXPLAINER_MODES_dict = {
    "GNNExplainer": "generate",                # generate | load
    "PGExplainer": "train",                 # train | test | load
    "GraphMask": "train",                   # train | test | load
    "SubGraphX": "generate",                   # generate | load
    "CF2": "generate",                         # generate | load
    "PGMExplainer": "generate",                # generate | load
    "XGNN": "generate",                        # generate | load
    "GNNInterpreter": "generate",          # generate | load
}


EXPLAINER_DATASET_BlockList = {"XGNN": {"Graph-SST5"},}

SKIP_COMBINATIONS = set()


batch_size = 16
gnn_model_loading_epoch = 5000
classifier_weight_decay = 1e-6

explainer_epoch = 2
explainer_learning_rate = 0.0001

save_root = "/data/cs.aau.dk/ey33jw/Meta_Explainer_on_Benchmarking"


for DataSet_name in RUN_DATASETS:

    print("\n")
    print("DATASET:", DataSet_name)

    loading_dataset = Load_Dataset_Module.load_dataset(dataset_name=DataSet_name, BATCH_SIZE=batch_size)

    (train_dataset, test_dataset, train_dataloader, test_dataloader, num_classes, entire_dataset) = loading_dataset()


    for GNN_name in RUN_GNNS:

        gnn_model_index = GNN_NAME_TO_INDEX_dict[GNN_name]

        print("\n")
        print("GNN:", GNN_name)

        loading_gnn_model = Create_and_Load_GNNS_Module.load_GNN_Models(gnn_model_index=gnn_model_index,
                                                                        dataset_name=DataSet_name,
                                                                        num_classes=num_classes,
                                                                        entire_dataset=entire_dataset,
                                                                        dropout_rate=
                                                                        all_parameters[gnn_model_index][DataSet_name][
                                                                            "dropout"],
                                                                        bias=
                                                                        all_parameters[gnn_model_index][DataSet_name][
                                                                            "bias"],
                                                                        weight_initializer=
                                                                        all_parameters[gnn_model_index][DataSet_name][
                                                                            "weight_initializer"],
                                                                        classifier_lr=
                                                                        all_parameters[gnn_model_index][DataSet_name][
                                                                            "lr"],
                                                                        act_fun="ReLu",
                                                                        gnn_model_loading_epoch=gnn_model_loading_epoch,
                                                                        classifier_weight_decay=classifier_weight_decay)

        GNN_Model_Optimizer, GNN_Model_Loaded = loading_gnn_model()
        GNN_Model_Loaded.eval()

        checkpoint_path = os.path.join(
            save_root,
            "Train_Models_on_Graph_Classification",
            "Experimental_Results",
            GNN_name,
            f"{GNN_name}_{DataSet_name}_{gnn_model_loading_epoch}.pt",
        )

        print("[CHECKPOINT] Path:", checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )
        saved_state = checkpoint["model_state_dict"]

        # Print available metadata without assuming its contents.
        print("[CHECKPOINT] Available fields:", list(checkpoint.keys()))

        for field in ("epoch", "loss", "train_loss", "accuracy", "test_accuracy"):
            value = checkpoint.get(field)

            if isinstance(value, (int, float, str)):
                print(f"[CHECKPOINT] {field}: {value}")
            elif torch.is_tensor(value) and value.numel() == 1:
                print(f"[CHECKPOINT] {field}: {value.item()}")

        for name, parameter in GNN_Model_Loaded.named_parameters():
            saved = saved_state[name].detach().cpu()
            loaded = parameter.detach().cpu()

            torch.testing.assert_close(
                loaded,
                saved,
                rtol=0,
                atol=0,
            )

            print(
                f"[CHECKPOINT MATCH] {name} | "
                f"saved_abs_max={saved.abs().max().item():.8g} | "
                f"loaded_abs_max={loaded.abs().max().item():.8g}"
            )

        print("[PASS] Loaded parameters exactly match the saved checkpoint.")

        del checkpoint, saved_state
        print("[CHECKPOINT DIAG] Parameters immediately after loading:")
        for name, parameter in GNN_Model_Loaded.named_parameters():
            print(
                name,
                "shape=", tuple(parameter.shape),
                "abs_max=", parameter.detach().abs().max().item(),
            )
        if VERIFY_GRAPHMASK_REFERENCE:
            if GNN_name != "GCN_plus_GAP_Model":
                raise ValueError("Reference verification currently supports GCN only.")

            from Meta_Explainer_on_Benchmarking.GraphMask_on_Graph_Classification.Script import (
                graphmask_on_graph_classification as GraphMask_Module)

            GraphMask_Module = reload(GraphMask_Module)

            adapter = GraphMask_Module.Hooking_GCN_plus_GAP_Model_for_GraphMask(gnn_model=GNN_Model_Loaded)

            device = next(GNN_Model_Loaded.parameters()).device
            checked_batches = 0

            with torch.no_grad():
                for batch_index, batch in enumerate(test_dataloader):
                    batch = batch.to(device)

                    original_hidden, original_pooling, original_classifier, original_prob = (GNN_Model_Loaded(batch, None))

                    reference = adapter.hooking_reference_forward(batch)

                    batch_size, max_nodes, _ = (
                        reference["layer_inputs"][0].shape
                    )

                    dtype = reference["layer_inputs"][0].dtype

                    open_gates = [
                        torch.ones(
                            batch_size,
                            max_nodes,
                            max_nodes,
                            device=device,
                            dtype=dtype,
                        )
                        for _ in GNN_Model_Loaded.GCN_Layers
                    ]

                    # Nonzero baselines verify that open gates exclude
                    # the replacement contribution completely.
                    baselines = [
                        torch.full(
                            (layer.conv_params.out_features,),
                            0.25,
                            device=device,
                            dtype=dtype,
                        )
                        for layer in GNN_Model_Loaded.GCN_Layers
                    ]

                    masked = adapter.hooking_masked_forward(batch, gates=open_gates, baselines=baselines,)

                    for layer_index, expected in enumerate(original_hidden):
                        torch.testing.assert_close(masked["hidden_outputs"][layer_index], expected, rtol=1e-5, atol=1e-6,)

                    for name, expected in (
                            ("pooling_output", original_pooling),
                            ("classifier_output", original_classifier),
                            ("probabilities", original_prob),
                    ):
                        torch.testing.assert_close(
                            masked[name],
                            expected,
                            rtol=1e-5,
                            atol=1e-6,
                        )

                    print(
                        f"[PASS] All-open GraphMask | "
                        f"batch={batch_index + 1} | "
                        f"max_probability_difference="
                        f"{(masked['probabilities'] - original_prob).abs().max().item():.8g}"
                    )

                    assert len(reference["layer_inputs"]) == len(GNN_Model_Loaded.GCN_Layers)
                    assert len(reference["hidden_outputs"]) == len(original_hidden)

                    # Compare every hidden layer.
                    for layer_index, (actual, expected) in enumerate(zip(reference["hidden_outputs"], original_hidden)):
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

                        # Verify the captured input dimensions.
                        layer_input = reference["layer_inputs"][layer_index]
                        expected_features = (GNN_Model_Loaded.GCN_Layers[layer_index].conv_params.in_features)

                        assert layer_input.shape[-1] == expected_features
                        assert not layer_input.requires_grad

                        # Later layers must receive the preceding layer's output.
                        if layer_index > 0:
                            torch.testing.assert_close(layer_input, original_hidden[layer_index - 1], rtol=1e-5,
                                                       atol=1e-6, )

                    for name, expected in (
                    ("pooling_output", original_pooling), ("classifier_output", original_classifier),
                    ("probabilities", original_prob),):
                        actual = reference[name]
                        assert torch.isfinite(actual).all().item()
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

                    max_difference = (reference["probabilities"] - original_prob).abs().max().item()

                    print(f"[PASS] {DataSet_name} | {GNN_name} | "
                          f"batch={batch_index + 1} | "
                          f"graphs={batch.num_graphs} | "
                          f"max_probability_difference={max_difference:.8g}")
                    print("Layer input shapes:",
                          [tuple(x.shape) for x in reference["layer_inputs"]], )

                    checked_batches += 1
                    if checked_batches == 3:
                        break

            if checked_batches == 0:
                raise RuntimeError("Verification loader contains no batches.")

            print(f"Reference verification passed for {checked_batches} batches.")
            # ============================================================
            # Verify baseline replacement and gradient flow.
            # IMPORTANT: This block is outside torch.no_grad().
            # ============================================================

            hooking = GraphMask_Module.Hooking_GCN_plus_GAP_Model_for_GraphMask(gnn_model=GNN_Model_Loaded)

            device = next(GNN_Model_Loaded.parameters()).device
            batch = next(iter(test_dataloader)).to(device)

            # Temporarily freeze GNN weights, preserving their current flags.
            parameters = list(GNN_Model_Loaded.parameters())
            original_requires_grad = [p.requires_grad for p in parameters]

            for parameter in parameters:
                parameter.requires_grad_(False)

            try:
                # --------------------------------------------------------
                # 1. Verify closed gates replace neighbor contributions.
                # --------------------------------------------------------
                with torch.no_grad():
                    adjacency, degree_norm, initial_x = (
                        GNN_Model_Loaded.computational_matrices(batch, None)
                    )

                    if adjacency.layout != torch.strided:
                        adjacency = adjacency.to_dense()

                    normalized_adj = torch.bmm(
                        degree_norm,
                        torch.bmm(adjacency, degree_norm),
                    )

                    diagonal = torch.eye(
                        normalized_adj.size(-1),
                        device=device,
                        dtype=torch.bool,
                    ).unsqueeze(0)

                    # First layer: close neighbor gates.
                    # Later layers: keep all gates open.
                    closed_gates = [
                        torch.zeros_like(normalized_adj)
                        if index == 0
                        else torch.ones_like(normalized_adj)
                        for index in range(len(GNN_Model_Loaded.GCN_Layers))
                    ]

                    baselines = [
                        torch.full(
                            (layer.conv_params.out_features,),
                            0.25,
                            device=device,
                            dtype=initial_x.dtype,
                        )
                        for layer in GNN_Model_Loaded.GCN_Layers
                    ]

                    closed_result = hooking.hooking_masked_forward(
                        batch,
                        gates=closed_gates,
                        baselines=baselines,
                    )

                    # Self-loops remain open in our implementation.
                    self_adj = torch.where(
                        diagonal,
                        normalized_adj,
                        torch.zeros_like(normalized_adj),
                    )
                    neighbor_adj = normalized_adj - self_adj

                    first_layer = GNN_Model_Loaded.GCN_Layers[0]

                    expected_first_hidden = first_layer.conv_params(
                        torch.bmm(self_adj, initial_x)
                    )

                    expected_first_hidden = (
                            expected_first_hidden
                            + neighbor_adj.sum(-1, keepdim=True)
                            * baselines[0].view(1, 1, -1)
                    )

                    expected_first_hidden = GNN_Model_Loaded.act_fun(
                        expected_first_hidden
                    )
                    expected_first_hidden = GNN_Model_Loaded.dropout(
                        expected_first_hidden
                    )

                    torch.testing.assert_close(
                        closed_result["hidden_outputs"][0],
                        expected_first_hidden,
                        rtol=1e-5,
                        atol=1e-6,
                    )

                    print(
                        "[PASS] Closed neighbor gates produce the expected "
                        "baseline replacement; self-loops remain open."
                    )
                    for layer_index, layer in enumerate(
                            GNN_Model_Loaded.GCN_Layers
                    ):
                        layer_input = reference["layer_inputs"][layer_index]
                        layer_output = reference["hidden_outputs"][layer_index]
                        weight = layer.conv_params.weight
                        bias = layer.conv_params.bias

                        print(
                            f"[MODEL DIAG] Layer {layer_index} | "
                            f"input_abs_max={layer_input.abs().max().item():.8g} | "
                            f"output_abs_max={layer_output.abs().max().item():.8g} | "
                            f"weight_abs_max={weight.abs().max().item():.8g} | "
                            f"output_nonzero={torch.count_nonzero(layer_output).item()}"
                        )

                        if bias is not None:
                            print(
                                f"[MODEL DIAG] Layer {layer_index} bias range:",
                                bias.min().item(),
                                bias.max().item(),
                            )

                    before_relu = GNN_Model_Loaded.ffn(
                        reference["pooling_output"]
                    )

                    print(
                        "[MODEL DIAG] Pooling absolute maximum:",
                        reference["pooling_output"].abs().max().item(),
                    )
                    print(
                        "[MODEL DIAG] Classifier BEFORE ReLU range:",
                        before_relu.min().item(),
                        before_relu.max().item(),
                    )
                    print(
                        "[MODEL DIAG] Classifier BEFORE ReLU, first 3 graphs:",
                        before_relu[:3].cpu().tolist(),
                    )

                # --------------------------------------------------------
                # 2. Verify gradients can reach gates and baselines.
                # --------------------------------------------------------

                # Partial gates exercise both retained and replaced paths.
                trainable_gates = [
                    torch.full_like(
                        normalized_adj,
                        0.5,
                        requires_grad=True,
                    )
                    for _ in GNN_Model_Loaded.GCN_Layers
                ]

                trainable_baselines = [
                    torch.full(
                        (layer.conv_params.out_features,),
                        0.25,
                        device=device,
                        dtype=initial_x.dtype,
                        requires_grad=True,
                    )
                    for layer in GNN_Model_Loaded.GCN_Layers
                ]

                gradient_result = hooking.hooking_masked_forward(
                    batch,
                    gates=trainable_gates,
                    baselines=trainable_baselines,
                )

                # Probe each layer's hidden output to avoid a saturated
                # classifier concealing the local intervention gradients.
                probe_loss = sum(
                    hidden.square().mean()
                    for hidden in gradient_result["hidden_outputs"]
                )

                gradients = torch.autograd.grad(
                    probe_loss,
                    trainable_gates + trainable_baselines,
                    allow_unused=True,
                    )

                layer_count = len(trainable_gates)
                gate_gradients = gradients[:layer_count]
                baseline_gradients = gradients[layer_count:]

                # Only original off-diagonal edges should carry gate gradients.
                neighbor_positions = (
                        (normalized_adj != 0) & ~diagonal
                )

                if not neighbor_positions.any().item():
                    raise RuntimeError(
                        "This batch has no neighbor edges to verify."
                    )

                for index, (gate_grad, baseline_grad) in enumerate(
                        zip(gate_gradients, baseline_gradients)
                ):
                    assert gate_grad is not None, (
                        f"Layer {index}: gate is disconnected from computation."
                    )
                    assert baseline_grad is not None, (
                        f"Layer {index}: baseline is disconnected."
                    )

                    assert torch.isfinite(gate_grad).all().item()
                    assert torch.isfinite(baseline_grad).all().item()

                    neighbor_gate_norm = (
                        gate_grad[neighbor_positions].abs().sum().item()
                    )
                    baseline_norm = baseline_grad.abs().sum().item()

                    # Diagonal gates are forced open, so their gradients are zero.
                    diagonal_grad = gate_grad.diagonal(dim1=-2, dim2=-1)
                    assert torch.count_nonzero(diagonal_grad).item() == 0

                    print(
                        f"[GRADIENT] layer={index} | "
                        f"neighbor_gate_gradient={neighbor_gate_norm:.8g} | "
                        f"baseline_gradient={baseline_norm:.8g}"
                    )

                    if neighbor_gate_norm == 0 or baseline_norm == 0:
                        raise RuntimeError(
                            f"Layer {index}: zero gradient under this probe. "
                            "Inspect activation saturation before proceeding."
                        )

                print(
                    "[PASS] Both layers have finite, nonzero gate and "
                    "baseline gradients."
                )

            finally:
                # Restore the model's original trainability flags.
                for parameter, requires_grad in zip(
                        parameters, original_requires_grad
                ):
                    parameter.requires_grad_(requires_grad)
            hooking = GraphMask_Module.Hooking_GCN_plus_GAP_Model_for_GraphMask(
                gnn_model=GNN_Model_Loaded
            )

            gate_module = GraphMask_Module.GCN_GraphMask_Gates(
                gnn_model=GNN_Model_Loaded,
                hidden_dim=64,
            )
            gate_module.train()

            batch = next(iter(test_dataloader)).to(device)
            reference = hooking.hooking_reference_forward(batch)

            gate_outputs = gate_module(
                batch,
                reference["layer_inputs"],
            )

            masked = hooking.hooking_masked_forward(
                batch,
                gates=gate_outputs["gate_matrices"],
                baselines=gate_module.baselines,
            )
            with torch.no_grad():
                original_classifier = reference["classifier_output"]
                masked_classifier = masked["classifier_output"]
                original_prob = reference["probabilities"]
                masked_prob = masked["probabilities"]

                print(
                    "[DIAG] Original classifier range:",
                    original_classifier.min().item(),
                    original_classifier.max().item(),
                )
                print(
                    "[DIAG] Masked classifier range:",
                    masked_classifier.min().item(),
                    masked_classifier.max().item(),
                )
                print(
                    "[DIAG] Maximum classifier difference:",
                    (masked_classifier - original_classifier)
                    .abs().max().item(),
                    )
                print(
                    "[DIAG] Maximum probability difference:",
                    (masked_prob - original_prob)
                    .abs().max().item(),
                    )

                print(
                    "[DIAG] Original probabilities, first 3 graphs:",
                    original_prob[:3].cpu().tolist(),
                )
                print(
                    "[DIAG] Masked probabilities, first 3 graphs:",
                    masked_prob[:3].cpu().tolist(),
                )

                neighbor_edges = (
                        batch.edge_index[0] != batch.edge_index[1]
                )

                for layer_index, edge_gates in enumerate(
                        gate_outputs["edge_gates"]
                ):
                    neighbor_gates = edge_gates[neighbor_edges]

                    print(
                        f"[DIAG] Layer {layer_index} neighbor gates | "
                        f"closed={(neighbor_gates == 0).sum().item()} | "
                        f"partial={((neighbor_gates > 0) & (neighbor_gates < 1)).sum().item()} | "
                        f"open={(neighbor_gates == 1).sum().item()}"
                    )

                    original_hidden = reference["hidden_outputs"][
                        layer_index
                    ]
                    masked_hidden = masked["hidden_outputs"][
                        layer_index
                    ]

                    print(
                        f"[DIAG] Layer {layer_index} | "
                        f"maximum hidden difference="
                        f"{(masked_hidden - original_hidden).abs().max().item():.8g}"
                    )

                labels = batch.y.reshape(-1)
                print(
                    "[DIAG] Original predictions:",
                    original_prob.argmax(dim=-1).cpu().tolist(),
                )
                print(
                    "[DIAG] Actual labels:",
                    labels.cpu().tolist(),
                )
            for layer_index, matrix in enumerate(
                    gate_outputs["gate_matrices"]
            ):
                scores = gate_outputs["edge_scores"][layer_index]

                assert scores.numel() == batch.edge_index.size(1)
                assert matrix.requires_grad
                assert torch.isfinite(matrix).all().item()
                assert ((matrix >= 0) & (matrix <= 1)).all().item()

                print(
                    f"[PASS] Gate mapping | layer={layer_index} | "
                    f"matrix_shape={tuple(matrix.shape)} | "
                    f"edge_scores={scores.numel()}"
                )

            assert torch.isfinite(masked["probabilities"]).all().item()

            print(
                "[PASS] Gate networks and baselines connect to "
                "hooking_masked_forward."
            )
            print(
                "Expected retained directed messages:",
                gate_outputs["expected_retained_messages"].item(),
            )
            # ============================================================
            # Verify gradients from the prediction-preservation objective.
            # No optimizer update is performed.
            # ============================================================

            gnn_parameters = list(GNN_Model_Loaded.parameters())
            original_flags = [
                parameter.requires_grad
                for parameter in gnn_parameters
            ]

            for parameter in gnn_parameters:
                parameter.requires_grad_(False)

            try:
                gate_module.train()

                # Recompute a fresh differentiable masking pass.
                gate_outputs = gate_module(
                    batch,
                    reference["layer_inputs"],
                )

                masked = hooking.hooking_masked_forward(
                    batch,
                    gates=gate_outputs["gate_matrices"],
                    baselines=gate_module.baselines,
                )

                # Match the entire original prediction distribution.
                reference_probabilities = (
                    reference["probabilities"].detach()
                )

                masked_log_probabilities = (
                    torch.nn.functional.log_softmax(
                        masked["classifier_output"],
                        dim=-1,
                    )
                )

                divergence = torch.nn.functional.kl_div(
                    masked_log_probabilities,
                    reference_probabilities,
                    reduction="batchmean",
                )

                # Expected retained directed layer-messages per graph.
                sparsity_loss = (
                        gate_outputs["expected_retained_messages"]
                        / batch.num_graphs
                )

                tolerance = 0.03
                constraint = divergence - tolerance

                # Fixed multiplier for verification only.
                # We will make it trainable in the training implementation.
                multiplier = divergence.new_tensor(1.0)

                objective = sparsity_loss + multiplier * constraint

                assert torch.isfinite(divergence).item()
                assert torch.isfinite(objective).item()

                gate_parameters = list(
                    gate_module.gate_networks.parameters()
                )
                baseline_parameters = list(gate_module.baselines)
                explainer_parameters = (
                        gate_parameters + baseline_parameters
                )

                # Inspect fidelity gradients separately so sparsity cannot
                # conceal a disconnected prediction-preservation path.
                fidelity_gradients = torch.autograd.grad(
                    divergence,
                    explainer_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )

                # Second gradient calculation uses the same forward graph.
                objective_gradients = torch.autograd.grad(
                    objective,
                    explainer_parameters,
                    allow_unused=True,
                )

                def checked_gradient_norm(gradients, label):
                    total = 0.0

                    for gradient in gradients:
                        if gradient is None:
                            raise RuntimeError(
                                f"{label}: a parameter is disconnected."
                            )

                        if not torch.isfinite(gradient).all().item():
                            raise RuntimeError(
                                f"{label}: nonfinite gradient detected."
                            )

                        total += gradient.abs().sum().item()

                    return total

                split = len(gate_parameters)

                fidelity_gate_norm = checked_gradient_norm(
                    fidelity_gradients[:split],
                    "Fidelity → gate networks",
                )
                fidelity_baseline_norm = checked_gradient_norm(
                    fidelity_gradients[split:],
                    "Fidelity → baselines",
                )

                objective_gate_norm = checked_gradient_norm(
                    objective_gradients[:split],
                    "Objective → gate networks",
                )
                objective_baseline_norm = checked_gradient_norm(
                    objective_gradients[split:],
                    "Objective → baselines",
                )

                print(
                    f"[LOSS] KL divergence={divergence.item():.8g} | "
                    f"expected_retained_per_graph="
                    f"{sparsity_loss.item():.8g} | "
                    f"constraint={constraint.item():.8g}"
                )

                print(
                    f"[FIDELITY GRADIENT] "
                    f"gate_networks={fidelity_gate_norm:.8g} | "
                    f"baselines={fidelity_baseline_norm:.8g}"
                )

                print(
                    f"[OBJECTIVE GRADIENT] "
                    f"gate_networks={objective_gate_norm:.8g} | "
                    f"baselines={objective_baseline_norm:.8g}"
                )

                if (
                        fidelity_gate_norm == 0
                        or fidelity_baseline_norm == 0
                ):
                    print(
                        "[CHECK NEEDED] Fidelity gradients are connected "
                        "but at least one group has zero gradient. "
                        "Inspect gate clipping and activation saturation."
                    )
                else:
                    print(
                        "[PASS] Prediction-preservation loss reaches "
                        "both gate networks and baselines."
                    )

                assert all(
                    not parameter.requires_grad
                    for parameter in gnn_parameters
                )

                print(
                    "[PASS] GNN parameters stayed frozen; "
                    "no optimizer updates were performed."
                )

            finally:
                for parameter, original_flag in zip(
                        gnn_parameters, original_flags
                ):
                    parameter.requires_grad_(original_flag)
            del adapter
            del GNN_Model_Loaded
            del GNN_Model_Optimizer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Skip the explainer loop for this GNN.
            continue

        for Explainer_Name in RUN_EXPLAINERS:
            restricted_datasets_for_the_explainer = EXPLAINER_DATASET_BlockList.get(Explainer_Name)
            if (restricted_datasets_for_the_explainer is not None
                    and DataSet_name in restricted_datasets_for_the_explainer):
                print(f"[SKIP] {Explainer_Name} is not supported for dataset {DataSet_name}.")
                continue

            if (DataSet_name, Explainer_Name, GNN_name) in SKIP_COMBINATIONS:
                print(f"[SKIP] Manual exclusion: "f"{DataSet_name} | {GNN_name} | {Explainer_Name}")
                continue
            if not running_file_checker.should_run(dataset_name=DataSet_name, gnn_name=GNN_name,
                                                   explainer_name=Explainer_Name, task_name="Graph_Classification",
                                                   verbose=True, explainer_mode=EXPLAINER_MODES_dict[Explainer_Name]):
                continue

            print("\nRUN:", f"dataset={DataSet_name}", f"gnn={GNN_name}", f"explainer={Explainer_Name}")

                ########################################################################################################

            if Explainer_Name == "GNNExplainer":
                from Meta_Explainer_on_Benchmarking.GNNExplainer_on_Graph_Classification.Script import \
                    gnnexplainer_on_graph_classification as GNNExplainer_Module
                GNNExplainer_Module = reload(GNNExplainer_Module)

                GNE_manager = GNNExplainer_Module.GNNExplainer_Manager(GNN_Model=GNN_Model_Loaded,
                                                                       dataset_name=DataSet_name,
                                                                       num_classes=num_classes,
                                                                       Exp_Epoch=explainer_epoch,
                                                                       Exp_lr=explainer_learning_rate,
                                                                       save_root=save_root)

                GNE_saved_data = GNE_manager(test_dataset=test_dataset, GNE_Mode=EXPLAINER_MODES_dict[Explainer_Name],
                                             binarization_threshold=0.5)

                print(GNN_name, " Model Explained by GNNExplainer", "\nTotal Runtime: ",
                      GNE_saved_data["total_runtime"], "\nAverage Explanation Runtime per Graph-Class: ",
                      GNE_saved_data["average_explanation_runtime_per_graph_class"],
                      "\nAverage Explanation Runtime for All Classes per Graph: ",
                      GNE_saved_data["average_explanation_runtime_all_classes_per_graph"], )

                ########################################################################################################

            elif Explainer_Name == "PGExplainer":
                from Meta_Explainer_on_Benchmarking.PGExplainer_on_Graph_Classification.Script import \
                    pgexplainer_on_graph_classification as PGExplainer_Module
                PGExplainer_Module = reload(PGExplainer_Module)
                pgex_dim = {"GCN_plus_GAP_Model": {"MUTAG": test_dataset[0].x.size(-1),
                                                   "NCI1": test_dataset[0].x.size(-1), "ENZYMES": 16, "Graph-SST5": 64,
                                                   "PROTEINS": 16, "IsCyclic": test_dataset[0].x.size(-1)},
                            "DGCNN_Model": {"MUTAG": 32, "NCI1": 32, "ENZYMES": 32,
                                            "Graph-SST5": 64, "PROTEINS": 32, "IsCyclic": 32},
                            "DIFFPOOL_Model": {"MUTAG": test_dataset[0].x.size(-1), "NCI1": 37,
                                               "ENZYMES": test_dataset[0].x.size(-1),
                                               "Graph-SST5": test_dataset[0].x.size(-1),
                                               "PROTEINS": test_dataset[0].x.size(-1),
                                               "IsCyclic": test_dataset[0].x.size(-1)},
                            "GIN_Model": {"MUTAG": test_dataset[0].x.size(-1), "NCI1": test_dataset[0].x.size(-1),
                                          "ENZYMES": 16, "Graph-SST5": 64, "PROTEINS": 16,
                                          "IsCyclic": test_dataset[0].x.size(-1)}}

                PGE_manager = PGExplainer_Module.PGExplainer_Manager(GNN_Model=GNN_Model_Loaded,
                                                                     dataset_name=DataSet_name, num_classes=num_classes,
                                                                     Exp_Epoch=explainer_epoch,
                                                                     Exp_lr=explainer_learning_rate,
                                                                     node_feat_dim=pgex_dim[GNN_name][DataSet_name],
                                                                     save_root=save_root)

                PGE_manager(test_dataset=test_dataset, PGE_Mode=EXPLAINER_MODES_dict[Explainer_Name],
                            binarization_threshold=0.5, explainer_training_dataset=test_dataloader)

                PGE_saved_data = PGE_manager(test_dataset=test_dataset, PGE_Mode="test", binarization_threshold=0.5)

                print(GNN_name, " Model Explained by PGExplainer", "\nTotal Runtime: ", PGE_saved_data["testing_runtime"],
                      "\nAverage Explanation Runtime per Graph-Class: ",
                      PGE_saved_data["average_explanation_runtime_per_graph_class"],
                      "\nAverage Explanation Runtime for All Classes per Graph: ",
                      PGE_saved_data["average_explanation_runtime_all_classes_per_graph"], )

                ########################################################################################################

            elif Explainer_Name == "GraphMask":
                from Meta_Explainer_on_Benchmarking.GraphMask_on_Graph_Classification.Script import \
                    graphmask_on_graph_classification as GraphMask_Module
                GraphMask_Module = reload(GraphMask_Module)

                graphmask_explainers_input_dims = {
                    "GCN_plus_GAP_Model": {
                        "MUTAG": test_dataset[0].x.size(-1),
                        "NCI1": test_dataset[0].x.size(-1),
                        "ENZYMES": 16,
                        "Graph-SST5": 64,
                        "PROTEINS": 16,
                        "IsCyclic": test_dataset[0].x.size(-1)
                    },
                    "DGCNN_Model": {
                        "MUTAG": 32, "NCI1": 32, "ENZYMES": 32,
                        "Graph-SST5": 64, "PROTEINS": 32, "IsCyclic": 32
                    },
                    "DIFFPOOL_Model": {
                        "MUTAG": test_dataset[0].x.size(-1),
                        "NCI1": test_dataset[0].x.size(-1),
                        "ENZYMES": test_dataset[0].x.size(-1),
                        "Graph-SST5": test_dataset[0].x.size(-1),
                        "PROTEINS": test_dataset[0].x.size(-1),
                        "IsCyclic": test_dataset[0].x.size(-1)
                    },
                    "GIN_Model": {
                        "MUTAG": test_dataset[0].x.size(-1),
                        "NCI1": test_dataset[0].x.size(-1),
                        "ENZYMES": 16,
                        "Graph-SST5": 64,
                        "PROTEINS": 16,
                        "IsCyclic": test_dataset[0].x.size(-1)
                    }
                }

                graphmask_explainers_hid_dims = {
                    "GCN_plus_GAP_Model": {
                        "MUTAG": test_dataset[0].x.size(-1),
                        "NCI1": test_dataset[0].x.size(-1),
                        "ENZYMES": test_dataset[0].x.size(-1),
                        "Graph-SST5": 64,
                        "PROTEINS": 16,
                        "IsCyclic": test_dataset[0].x.size(-1)
                    },
                    "DGCNN_Model": {
                        "MUTAG": test_dataset[0].x.size(-1),
                        "NCI1": test_dataset[0].x.size(-1),
                        "ENZYMES": test_dataset[0].x.size(-1),
                        "Graph-SST5": 64,
                        "PROTEINS": 32,
                        "IsCyclic": test_dataset[0].x.size(-1)
                    },
                    "DIFFPOOL_Model": {
                        "MUTAG": test_dataset[0].x.size(-1),
                        "NCI1": test_dataset[0].x.size(-1),
                        "ENZYMES": test_dataset[0].x.size(-1),
                        "Graph-SST5": 64,
                        "PROTEINS": test_dataset[0].x.size(-1),
                        "IsCyclic": test_dataset[0].x.size(-1)
                    },
                    "GIN_Model": {
                        "MUTAG": test_dataset[0].x.size(-1),
                        "NCI1": test_dataset[0].x.size(-1),
                        "ENZYMES": test_dataset[0].x.size(-1),
                        "Graph-SST5": 64,
                        "PROTEINS": 16,
                        "IsCyclic": test_dataset[0].x.size(-1)
                    }
                }

                GM_manager = GraphMask_Module.GraphMask_Manager(GNN_Model=GNN_Model_Loaded, dataset_name=DataSet_name,
                                                                num_classes=num_classes, Exp_Epoch=explainer_epoch,
                                                                Exp_lr=0.001, explainer_input_dim=
                                                                graphmask_explainers_input_dims[GNN_name][DataSet_name],
                                                                explainer_hid_dim=
                                                                graphmask_explainers_hid_dims[GNN_name][DataSet_name],
                                                                save_root=save_root)

                GM_Mode = EXPLAINER_MODES_dict[Explainer_Name]
                GM_saved_data = GM_manager(test_dataset=test_dataset, GM_Mode=GM_Mode,
                                           explainer_training_dataset=test_dataloader)

                if GM_Mode == "test" and GNN_Model_Loaded.__class__.__name__ == "DIFFPOOL_Model":

                    for class_index, masks in GM_saved_data["binary_edge_explanations"].items():

                        total_edges = sum(mask.numel() for mask in masks)
                        selected_edges = sum(mask.sum().item() for mask in masks)

                        print(
                            f"Class {class_index}: "
                            f"graphs={len(masks)}, "
                            f"selected_edges={selected_edges}/{total_edges}, "
                            f"selection_ratio={selected_edges / max(total_edges, 1):.4f}"
                        )




                if GM_Mode == "train":
                    print(GNN_name, "Model Explained by GraphMask", "\nTraining Runtime:",
                          GM_saved_data["training_runtime"])
                elif GM_Mode in ("test", "load"):
                    print(GNN_name, "Model Explained by GraphMask", "\nTesting Runtime:",
                          GM_saved_data["testing_runtime"])

                ########################################################################################################

            elif Explainer_Name == "SubGraphX":
                from Meta_Explainer_on_Benchmarking.SubGraphX_on_Graph_Classification.Script import \
                    subgraphx_on_graph_classification as SubGraphX_Module
                SubGraphX_Module = reload(SubGraphX_Module)

                SGX_manager = SubGraphX_Module.SubGraphX_Manager(GNN_Model=GNN_Model_Loaded, dataset_name=DataSet_name,
                                                                 num_classes=num_classes, num_hops=2, rollout_count=20,
                                                                 min_children_threshold=5, ubc1_c_coef=10.0,
                                                                 expand_count_threshold=5, high2low=True,
                                                                 sample_num=100, save_root=save_root)

                SGX_saved_data = SGX_manager(test_dataset=test_dataset, SGX_Mode=EXPLAINER_MODES_dict[Explainer_Name])

                print(GNN_name, " Model Explained by SubGraphX", "\nTotal Runtime: ", SGX_saved_data["total_runtime"],
                      "\nAverage Explanation Runtime per Graph-Class: ",
                      SGX_saved_data["average_explanation_runtime_per_graph_class"],
                      "\nAverage Explanation Runtime for All Classes per Graph: ",
                      SGX_saved_data["average_explanation_runtime_all_classes_per_graph"], )

                ########################################################################################################

            elif Explainer_Name == "CF2":
                from Meta_Explainer_on_Benchmarking.CF2_on_Graph_Classification.Script import \
                    cf2_on_graph_classification as CF2_Module
                CF2_Module = reload(CF2_Module)

                CF2_manager = CF2_Module.CF2_Manager(GNN_Model=GNN_Model_Loaded, dataset_name=DataSet_name,
                                                     num_classes=num_classes, explainer_epochs=explainer_epoch,
                                                     input_dim=test_dataset[0].x[0].size()[0],
                                                     hid_dim=test_dataset[0].x[0].size()[0], output_dim=num_classes,
                                                     save_root=save_root)

                CF2_saved_data = CF2_manager(test_dataset=test_dataset, CF2_Mode=EXPLAINER_MODES_dict[Explainer_Name],
                                             binarization_threshold=0.5)

                print(GNN_name, " Model Explained by CF2", "\nTotal Runtime: ", CF2_saved_data["total_runtime"],
                      "\nAverage Explanation Runtime per Graph-Class: ",
                      CF2_saved_data["average_explanation_runtime_per_graph_class"],
                      "\nAverage Explanation Runtime for All Classes per Graph: ",
                      CF2_saved_data["average_explanation_runtime_all_classes_per_graph"], )

                ########################################################################################################

            elif Explainer_Name == "PGMExplainer":
                from Meta_Explainer_on_Benchmarking.PGMExplainer_on_Graph_Classification.Script import \
                    pgmexplainer_on_graph_classification as PGMExplainer_Module
                PGMExplainer_Module = reload(PGMExplainer_Module)

                PGM_manager = PGMExplainer_Module.PGMExplainer_Manager(GNN_Model=GNN_Model_Loaded,
                                                                       dataset_name=DataSet_name,
                                                                       num_classes=num_classes,
                                                                       perturb_feature_list=[None], perturb_mode="mean",
                                                                       perturb_indicator="abs",
                                                                       num_samples_rule="num_nodes",
                                                                       noise_offset_percentage=50, top_node=5,
                                                                       p_value_threshold=0.05, save_root=save_root)

                PGM_saved_data = PGM_manager(PGM_Mode=EXPLAINER_MODES_dict[Explainer_Name], test_dataset=test_dataset)

                print(GNN_name, " Model Explained by PGMExplainer", "\nTotal Runtime: ",
                      PGM_saved_data["total_runtime"], "\nAverage Explanation Runtime per Graph-Class: ",
                      PGM_saved_data["average_explanation_runtime_per_graph_class"],
                      "\nAverage Explanation Runtime for All Classes per Graph: ",
                      PGM_saved_data["average_explanation_runtime_all_classes_per_graph"], )

                ########################################################################################################

            elif Explainer_Name == "XGNN":
                from Meta_Explainer_on_Benchmarking.XGNN_on_Graph_Classification.Script import \
                    xgnn_on_graph_classification as XGNN_Module
                XGNN_Module = reload(XGNN_Module)

                datasets_max_number_of_nodes = {
                    "MUTAG": 28,
                    "NCI1": 111,
                    "PROTEINS": 620,
                    "Graph-SST5": 56,
                    "ENZYMES": 126,
                    "IsCyclic": 40
                }

                XGNN_manager = XGNN_Module.XGNN_Manager(GNN_Model=GNN_Model_Loaded, dataset_name=DataSet_name,
                                                        num_classes=num_classes, explainer_epochs=explainer_epoch,
                                                        max_generation_iterations=10,
                                                        num_node_features=test_dataset[0].x.size(-1),
                                                        max_number_of_nodes=datasets_max_number_of_nodes[DataSet_name],
                                                        random_start=True, rollout_count=10, hyp_for_rollout=1,
                                                        hyp_for_rules=2, dropout_rate=0.5, explainer_lr=0.01, b1=0.9,
                                                        b2=0.999, weight_decay=5e-4, save_root=save_root,
                                                        random_seed=42)

                XGNN_Mode = EXPLAINER_MODES_dict[Explainer_Name]
                if XGNN_Mode == "generate":
                    XGNN_saved_data = XGNN_manager(XGNN_Mode="generate", reference_dataset=test_dataset)
                elif XGNN_Mode == "load":
                    XGNN_saved_data = XGNN_manager(XGNN_Mode="load")
                else:
                    raise ValueError("XGNN_Mode must be 'generate' or 'load'.")

                print(GNN_name, " Model Explained by XGNN", "\nTotal Runtime: ", XGNN_saved_data["total_runtime"],
                      "\nAverage Explanation Runtime per Class: ",
                      XGNN_saved_data["average_explanation_runtime_per_class"], )

                ########################################################################################################

            elif Explainer_Name == "GNNInterpreter":
                from Meta_Explainer_on_Benchmarking.GNNInterpreter_on_Graph_Classification.Script import \
                    gnninterpreter_on_graph_classification as GNNInterpreter_Module
                GNNInterpreter_Module = reload(GNNInterpreter_Module)
                explanations_size = GNNInterpreter_Module.get_averaged_explanation_sizes()
                # explanations_size["Graph-SST5"]["DIFFPOOL_Model"][4] = 2
                GNNI_max_nodes = explanations_size[DataSet_name][GNN_name]

                if not GNNI_max_nodes:
                    print(f"[SKIP] GNNInterpreter has no explanation-size configuration for "
                          f"{DataSet_name} | {GNN_name}.")
                    continue

                GNNInt_manager = GNNInterpreter_Module.GNNInterpreter_Manager(GNN_Model=GNN_Model_Loaded,
                                                                              dataset_name=DataSet_name,
                                                                              num_classes=num_classes,
                                                                              explanation_epochs=explainer_epoch,
                                                                              max_nodes=GNNI_max_nodes,
                                                                              num_node_classes=test_dataset[0].x.size(
                                                                                  -1), num_edge_classes=None,
                                                                              temperature=0.15, learning_node_feat=True,
                                                                              learning_edge_feat=False, budget=10,
                                                                              budget_order=2, budget_beta=1,
                                                                              target_probability_min=0.9,
                                                                              batch_size_for_same_sized_graphs=1,
                                                                              save_root=save_root, random_seed=42)

                GNNInt_saved_data = GNNInt_manager(GNNI_Mode=EXPLAINER_MODES_dict[Explainer_Name],
                                                   test_dataset=test_dataset)

                print(GNN_name, " Model Explained by GNNInterpreter", "\nTotal Runtime: ",
                      GNNInt_saved_data["total_runtime"], "\nAverage Explanation Runtime per Class: ",
                      GNNInt_saved_data["average_explanation_runtime_per_class"],)

        del GNN_Model_Loaded
        del GNN_Model_Optimizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()