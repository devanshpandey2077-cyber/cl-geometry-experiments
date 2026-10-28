"""Random tiny real backends; never scientific evidence for H2."""
import os
import unittest
import torch
from h2.engine import forward_model
from h2.lora import inject_lora

try:
    from transformers import Qwen3Config,Qwen3ForCausalLM,LlamaConfig,LlamaForCausalLM
    BACKEND_ERROR=None
except (ImportError,RuntimeError) as error:
    BACKEND_ERROR=str(error)
    if os.environ.get("H2_REQUIRE_BACKEND_TESTS")=="1": raise

@unittest.skipIf(BACKEND_ERROR is not None,"Matching Transformers/Qwen3 runtime unavailable locally")
class ActualBackendTests(unittest.TestCase):
    def check_model(self,config_class,model_class):
        torch.manual_seed(17)
        config=config_class(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                            num_attention_heads=4,num_key_value_heads=2,head_dim=8,pad_token_id=0)
        config._attn_implementation="eager"
        model=model_class(config).requires_grad_(False)
        batch={"input_ids":torch.tensor([[0,4,8,12],[7,9,4,2]]),
               "attention_mask":torch.tensor([[0,1,1,1],[1,1,1,1]]),
               "position_ids":torch.tensor([[0,0,1,2],[0,1,2,3]])}
        model.eval()
        with torch.no_grad():baseline=forward_model(model,batch).logits.clone()
        inject_lora(model,dict(rank=2,alpha=4,dropout=.05,targets=["q_proj","k_proj","v_proj","o_proj"]))
        with torch.no_grad():
            out=forward_model(model,batch,hidden=True)
            torch.testing.assert_close(baseline,out.logits)
            self.assertEqual(len(out.hidden_states),3)
            self.assertEqual(out.logits.shape,(2,1,128))
        frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        model.train(); out=forward_model(model,batch)
        torch.nn.functional.cross_entropy(out.logits[:,-1,:],torch.tensor([65,66])).backward()
        optimizer.step()
        self.assertTrue(any(p.abs().sum()>0 for n,p in model.named_parameters() if "lora_B" in n))
        for n,p in model.named_parameters():
            if n in frozen:self.assertTrue(torch.equal(p,frozen[n]))
    def test_real_tiny_qwen3(self):self.check_model(Qwen3Config,Qwen3ForCausalLM)
    def test_real_tiny_llama(self):self.check_model(LlamaConfig,LlamaForCausalLM)
