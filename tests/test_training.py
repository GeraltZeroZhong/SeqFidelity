"""Analytic normalized objectives and grouped validation error."""

import numpy as np
import pandas as pd
import pytest
import torch

from seqfidelity.training import TrainConfig, loss_for, macro_mse


def loss_inputs():
    output = {"delta": torch.tensor([3., -1.], requires_grad=True),
              "y_ref": torch.tensor([2., 3.], requires_grad=True),
              "y_alt": torch.tensor([5., 2.], requires_grad=True)}
    batch = {"delta": torch.tensor([1., 1.]),
             "activity": torch.tensor([[0., 1.], [float("nan"), 0.]])}
    return output, batch


def test_joint_objective_normalizes_each_label_type_and_masks_missing_activity():
    output, batch = loss_inputs()
    config = TrainConfig(device="cpu", amp=False, objective="delta_activity", absolute_weight=.25)
    loss, delta_loss, activity_loss = loss_for(output, batch, config, delta_scale=2., activity_scale=2.)
    assert delta_loss.item() == 1.
    assert activity_loss.item() == 2.
    assert loss.item() == 1.5
    loss.backward()
    torch.testing.assert_close(output["delta"].grad, torch.tensor([.5, -.5]), rtol=0, atol=0)
    assert output["y_ref"].grad[1].item() == 0.
    assert output["y_ref"].grad[0].item() > 0.


def test_activity_objective_has_no_delta_loss_gradient():
    output, batch = loss_inputs()
    config = TrainConfig(device="cpu", amp=False, objective="activity")
    loss, delta_loss, activity_loss = loss_for(output, batch, config, delta_scale=2., activity_scale=2.)
    assert delta_loss.item() == 1.
    assert loss.item() == activity_loss.item() == 2.
    loss.backward()
    assert output["delta"].grad is None


def test_missing_activity_contributes_zero_to_joint_objective():
    output, batch = loss_inputs()
    batch["activity"].fill_(float("nan"))
    config = TrainConfig(device="cpu", amp=False, objective="delta_activity")
    loss, delta_loss, activity_loss = loss_for(output, batch, config, delta_scale=2., activity_scale=2.)
    assert activity_loss.item() == 0.
    assert loss.item() == delta_loss.item() == 1.
    loss.backward()
    assert torch.equal(output["y_ref"].grad, torch.zeros(2))
    assert torch.equal(output["y_alt"].grad, torch.zeros(2))
    config.objective = "activity"
    with pytest.raises(ValueError, match="activity labels"):
        loss_for(output, batch, config, delta_scale=2., activity_scale=2.)


def test_validation_error_uses_group_means():
    frame = pd.DataFrame({"delta": [1., 1., 4.], "locus_id": ["group_a", "group_a", "group_b"]})
    predictions = np.array([0., 0., 4.])
    assert macro_mse(frame, predictions) == .5
    duplicated = pd.concat([frame, frame.iloc[:2]], ignore_index=True)
    assert macro_mse(duplicated, np.array([0., 0., 4., 0., 0.])) == .5


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_validation_error_rejects_nonfinite_predictions(value):
    frame = pd.DataFrame({"delta": [1.], "locus_id": ["group_a"]})
    with pytest.raises(FloatingPointError, match="Non-finite"):
        macro_mse(frame, [value])


def test_tiny_cpu_training_checkpoint_round_trip_and_completed_resume(tmp_path):
    import json

    from seqfidelity.model import ModelConfig
    from seqfidelity.training import load_model, predict, train_model

    frame = pd.DataFrame({
        "ref_seq": ["ACGTAC", "CGTACG", "GTACGT", "TACGTA"],
        "alt_seq": ["ATGTAC", "CATACG", "GCACGT", "TTCGTA"],
        "delta": [1., -1., 2., -2.],
        "y_ref": [0., 2., 0., 2.], "y_alt": [1., 1., 2., 0.],
        "pair_id": [f"pair_{index}" for index in range(4)],
        "locus_id": ["group_a", "group_a", "group_b", "group_b"],
    })
    model_config = ModelConfig(mode="fixed", dim=8, heads=2, depth=1,
                               encoder_depth=1, decoder_depth=1, expansion=2, compression=2)
    config = TrainConfig(seed=0, device="cpu", amp=False, epochs=1, min_epochs=1,
                         patience=2, batch_size=2)
    directory = tmp_path / "training"
    model = train_model(model_config, config, frame, frame, directory)
    before = predict(model, frame, config)
    restored, restored_config = load_model(directory / "checkpoints" / "best.pt", device="cpu")
    after = predict(restored, frame, restored_config)
    for column in ("delta_pred", "y_ref_pred", "y_alt_pred"):
        np.testing.assert_array_equal(before[column], after[column])
        assert np.isfinite(before[column]).all()
    assert before.pair_id.tolist() == frame.pair_id.tolist()
    np.testing.assert_allclose(before.delta_pred, before.y_alt_pred - before.y_ref_pred)
    history = json.loads((directory / "history.json").read_text())
    assert len(history) == 1
    assert history[0]["pairs_seen"] == len(frame)
    selected_weights = {name: value.clone() for name, value in model.state_dict().items()}
    resumed = train_model(model_config, config, frame, frame, directory)
    assert json.loads((directory / "history.json").read_text()) == history
    assert all(torch.equal(value, resumed.state_dict()[name]) for name, value in selected_weights.items())
