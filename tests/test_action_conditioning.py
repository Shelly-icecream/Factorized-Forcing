import ast
from pathlib import Path
import unittest

import torch
from torch import nn
from utils.action_conditioning import enhance_action_projection


class ActionConditioningTests(unittest.TestCase):
    def test_projection_padding_missing_and_zero(self):
        context = torch.tensor([[[1., 2.], [0., 0.]], [[3., 4.], [0., 0.]]])
        actions = [torch.tensor([[1., 0.], [1., 0.]]), None]
        result = enhance_action_projection(context, actions, .5)
        torch.testing.assert_close(result[0, 0], torch.tensor([1.5, 2.]))
        torch.testing.assert_close(result[1], context[1])
        torch.testing.assert_close(result[:, 1], context[:, 1])
        self.assertIs(enhance_action_projection(context, actions, 0), context)
        torch.testing.assert_close(enhance_action_projection(context, [torch.zeros(1, 2), None], 1), context)
        for scale in (-1, float('nan')):
            with self.assertRaises(ValueError):
                enhance_action_projection(context, actions, scale)

    def test_encoder_keeps_original_and_routes_action_copy(self):
        from unittest.mock import patch
        tree = ast.parse(Path('utils/wan_wrapper.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WanTextEncoder')
        namespace = {'torch': torch, 'List': list}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<encoder>', 'exec'), namespace)
        class Encoder(namespace['WanTextEncoder']):
            def __init__(self):
                nn.Module.__init__(self)
                self.generator_action_scale = .5
                self.action_parser_version = 'test'
                self.text_encoder = lambda ids, mask: torch.tensor([[[1., 2.], [2., 0.], [9., 9.]]]).expand(ids.shape[0], -1, -1).clone()
                self.tokenizer = lambda texts, **kwargs: (torch.ones(len(texts), 3, dtype=torch.long), torch.tensor([[1, 1, 0]]).expand(len(texts), -1))
            @property
            def device(self):
                return torch.device('cpu')
        with patch('utils.action_conditioning.extract_action', side_effect=lambda text, version: 'running' if text == 'full' else ''):
            result = Encoder()(['full', 'without action'])
        self.assertEqual(result['prompt_embeds'][0, 0, 0].item(), 1.)
        self.assertFalse(torch.equal(result['generator_prompt_embeds'][0], result['prompt_embeds'][0]))
        torch.testing.assert_close(result['generator_prompt_embeds'][1], result['prompt_embeds'][1])
        self.assertEqual(result['generator_prompt_embeds'][:, 2].abs().sum().item(), 0.)

    def test_lora_covers_self_and_cross_attention(self):
        # Load the installer alone without importing CUDA/Wan dependencies.
        tree = ast.parse(Path('utils/wan_wrapper.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'add_lora_adapters')
        class Adapter(nn.Module):
            def __init__(self, linear, *args):
                super().__init__()
                self.base = linear
        namespace = {'nn': nn, 'LoRALinear': Adapter}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<installer>', 'exec'), namespace)
        model = nn.Module()
        for name in ('self_attn', 'cross_attn'):
            branch = nn.Module()
            branch.q = nn.Linear(2, 2)
            model.add_module(name, branch)
        namespace['add_lora_adapters'](model, 1, 1, 0, ('q',))
        self.assertIsInstance(model.self_attn.q, Adapter)
        self.assertIsInstance(model.cross_attn.q, Adapter)


if __name__ == '__main__':
    unittest.main()
