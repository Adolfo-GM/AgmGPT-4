"""Local AgmGPT-4 inference using its training tokenizer and best checkpoint."""
import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import GPT,GPTConfig
ROOT=Path(__file__).resolve().parent
class Chat:
    def __init__(self,device=None,variant="completion"):
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if variant not in ('completion','instruct'):raise ValueError('Unknown model')
        names=['base.pt'] if variant=='completion' else ['instruct.pt']
        path=next((ROOT/'out'/name for name in names if (ROOT/'out'/name).exists()),ROOT/'out'/names[0])
        self.pretrained=variant=='completion';self.checkpoint=path.name
        if not path.exists():raise RuntimeError('No AgmGPT-4 checkpoint yet. Place both release checkpoints in out/.')
        c=torch.load(path,map_location='cpu',weights_only=True);self.model=GPT(GPTConfig(**c['model_args'])).to(self.device);self.model.load_state_dict(c['model']);self.model.eval()
        self.tokenizer=Tokenizer.from_file(str(ROOT/'tokenizer.json'))
    @torch.inference_mode()
    def reply(self,text,history=None,max_new_tokens=256,on_token=None):
        history=history or [];context=''.join(f'<|user|> {h["user"]} <|ai|> {h["ai"]} <|eos|>\n\n' for h in history[-3:]);ids=self.tokenizer.encode((text if self.pretrained else context+f'<|user|> {text} <|ai|>')).ids;x=torch.tensor([ids],dtype=torch.long,device=self.device)
        result=[];stop={self.tokenizer.token_to_id(t) for t in ['<|eos|>','<|user|>']}
        for _ in range(max_new_tokens):
            logits,_=self.model(x[:,-self.model.config.block_size:]);logits=logits[:,-1,:]/0.8;values,_=torch.topk(logits,min(40,logits.size(-1)));logits[logits<values[:,-1,None]]=-float('inf');token=torch.multinomial(torch.softmax(logits,dim=-1),1)
            if token.item() in stop:break
            result.append(token.item());x=torch.cat([x,token],dim=1)
            if on_token is not None:
                decoded=self.tokenizer.decode(result,skip_special_tokens=True)
                if not decoded.endswith('\ufffd'):on_token(decoded)
        return self.tokenizer.decode(result,skip_special_tokens=True).strip()
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',nargs='?',default='chat');p.add_argument('--prompt',default='Explain why the sky is blue.');p.add_argument('--max-new-tokens',type=int,default=256);args=p.parse_args();chat=Chat()
    if args.mode=='sample':print(chat.reply(args.prompt,max_new_tokens=args.max_new_tokens));return
    print('AgmGPT-4 experimental local model. Type exit to quit.');history=[]
    while True:
        try:text=input('You: ')
        except EOFError:break
        if text.lower()=='exit':break
        reply=chat.reply(text,history,args.max_new_tokens);print('AgmGPT-4:',reply);history.append(dict(user=text,ai=reply))
if __name__=='__main__':main()

