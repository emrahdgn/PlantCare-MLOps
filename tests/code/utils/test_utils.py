import os
import tempfile
from argparse import Namespace

import numpy as np
import pytest
import torch
from torchvision.models.efficientnet import efficientnet_b0

import plantcare.utils.utils as utils


def test_set_seeds():
    # Check that set_seeds function sets the same seed for all random number generators
    SEED = 42

    # Set the seed
    utils.set_seeds(SEED)

    # Check that random numbers generated by numpy and torch are the same
    np_a = np.random.randn(2, 3)
    torch_a = torch.randn(2, 3).numpy()

    # Reset the seed and check that random numbers are the same as before
    utils.set_seeds(SEED)
    np_x = np.random.randn(2, 3)
    torch_x = torch.randn(2, 3).numpy()
    assert np.array_equal(np_x, np_a)
    assert np.array_equal(torch_x, torch_a)

    # Check that changing the seed generates different random numbers
    utils.set_seeds(SEED + 1)
    np_c = np.random.randn(2, 3)
    torch_c = torch.randn(2, 3).numpy()
    assert not np.array_equal(np_c, np_a)
    assert not np.array_equal(torch_c, torch_a)


def test_add_dict_to_dict():
    # Check that add_dict_to_dict function adds a nested dictionary to another dictionary
    dict1 = {"a": 1, "b": {"c": 2}}
    dict2 = {"b": {"d": 3}, "e": 4}
    utils.add_dict_to_dict(dict1, dict2)
    assert dict1 == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}


def test_flatten_dict():
    # Check that flatten_dict function flattens a nested dictionary
    d = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    flattened_d = utils.flatten_dict(d)
    assert flattened_d == {"a": 1, "b_c": 2, "b_d_e": 3}


def test_dict_to_string():
    d = {"a": 1, "b": 2, "c": 3}
    expected_output = "a: 1\nb: 2\nc: 3"
    assert utils.dict_to_string(d) == expected_output


def test_save_and_read_json():
    data = {"name": "John", "age": 30, "city": "New York"}
    file_path = "test.json"
    utils.save_json(data, file_path)
    assert os.path.isfile(file_path) == True
    assert utils.read_json(file_path) == data
    os.remove(file_path)


def test_set_parameter_requires_grad():
    # Test that all parameters in model have requires_grad set to False
    model = torch.nn.Linear(10, 5)
    utils.set_parameter_requires_grad(model, requires_grad=False)
    for param in model.parameters():
        assert param.requires_grad == False

    # Test that all parameters in model have requires_grad set to True
    utils.set_parameter_requires_grad(model, requires_grad=True)
    for param in model.parameters():
        assert param.requires_grad == True


def test_make_trainable_only_classifier():
    # Test that all parameters in model except classifier have requires_grad set to False
    model = efficientnet_b0()
    utils.set_parameter_requires_grad(model, requires_grad=True)
    utils.make_trainable_only_classifier(model)

    for name, param in model.named_parameters():
        if "classifier" in name:
            assert param.requires_grad == True
        else:
            assert param.requires_grad == False


def test_is_model_on_cuda():
    # Test that is_model_on_cuda returns correct value for CPU and GPU
    model = torch.nn.Linear(10, 5)
    assert utils.is_model_on_cuda(model) == False

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        assert utils.is_model_on_cuda(model) == True
    else:
        assert utils.is_model_on_cuda(model) == False


def test_convert_namespace_to_dict():
    # Test that convert_namespace_to_dict returns dictionary with same values as namespace object
    ns_obj = Namespace(a=1, b=2, c=Namespace(d=3, e=4))
    ns_dict = utils.convert_namespace_to_dict(ns_obj)
    assert ns_dict == {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}


def test_convert_dict_to_namespace():
    # Test that convert_dict_to_namespace returns namespace object with same values as dictionary
    dictionary = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
    ns_obj = utils.convert_dict_to_namespace(dictionary)
    assert ns_obj.a == 1
    assert ns_obj.b == 2
    assert ns_obj.c.d == 3
    assert ns_obj.c.e == 4


@pytest.fixture
def test_data():
    return {"foo": "bar", "baz": 123}


def test_save_and_load_object(test_data):
    # Save the test data to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        file_path = f.name
        utils.save_object(file_path, test_data)

    # Load the test data from the temporary file
    loaded_data = utils.load_object(file_path)

    # Assert that the loaded data is the same as the original test data
    assert loaded_data == test_data

    # Clean up the temporary file
    os.remove(file_path)


def test_load_non_existent_file():
    # Attempt to load an object from a non-existent file
    with pytest.raises(FileNotFoundError):
        utils.load_object("path/to/non-existent/file")