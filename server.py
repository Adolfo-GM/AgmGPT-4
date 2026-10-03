"""AgmGPT-4 local inference server; no external model calls."""
import argparse,json,threading
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import torch
from inference import Chat
ROOT=Path(__file__).resolve().parent;models={};lock=threading.Lock()
torch.set_num_threads(2);torch.set_num_interop_threads(1)
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT/'public'),**kwargs)
    def log_message(self,*args):pass
    def respond(self,code,data):
        body=json.dumps(data).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
    def do_GET(self):
        if urlsplit(self.path).path not in ('/','/index.html'):self.send_error(404);return
        super().do_GET()
    def do_POST(self):
        if self.path!='/api/chat':self.send_error(404);return
        origin=self.headers.get('Origin')
        if origin and urlsplit(origin).netloc not in (self.headers.get('Host'),self.headers.get('X-Forwarded-Host')):self.send_error(403);return
        if not lock.acquire(blocking=False):self.respond(409,dict(error='A response is generating. Please wait.'));return
        streaming=False
        def event(data):self.wfile.write((json.dumps(data)+'\n').encode());self.wfile.flush()
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=32000:raise ValueError('Invalid message size')
            data=json.loads(self.rfile.read(size));text=data.get('message');variant=data.get('model','completion')
            if not isinstance(text,str) or not text.strip() or len(text)>4000:raise ValueError('Enter a message of 1–4000 characters')
            if variant not in ('completion','instruct'):raise ValueError('Unknown model')
            history=data.get('history',[])
            if not isinstance(history,list):raise ValueError('Invalid history')
            history=[h for h in history[-3:] if isinstance(h,dict) and isinstance(h.get('user'),str) and isinstance(h.get('ai'),str)]
            self.send_response(200);self.send_header('Content-Type','application/x-ndjson; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('X-Accel-Buffering','no');self.send_header('Connection','close');self.end_headers();self.close_connection=True;streaming=True;event(dict(type='loading'))
            if variant not in models:models[variant]=Chat(device='cpu',variant=variant)
            model=models[variant];reply=model.reply(text,history,max_new_tokens=160,on_token=lambda text:event(dict(type='token',text=text)))
            event(dict(type='done',reply=reply or '(No text generated.)',checkpoint=model.checkpoint,pretrained=model.pretrained))
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception:
            if streaming:event(dict(type='error',error='Could not generate a response. Please retry.'))
            else:self.respond(400,dict(error='Invalid request'))
        finally:lock.release()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    print(f'AgmGPT-4: http://{args.host}:{args.port}',flush=True)
    ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()
