import sys, os, base64, nbformat
from nbclient import NotebookClient
src = sys.argv[1]; name = os.path.basename(src).replace('.ipynb','')
nb = nbformat.read(src, as_version=4)
NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': os.path.dirname(os.path.abspath(src))}}).execute()
out = f'run/{name}'; os.makedirs(out, exist_ok=True)
nbformat.write(nb, f'{out}/{name}.ipynb')
k=0
for i,c in enumerate(nb.cells):
    for o in c.get('outputs', []):
        d = o.get('data', {})
        if 'image/png' in d:
            k+=1; open(f'{out}/cell{i:02d}_{k}.png','wb').write(base64.b64decode(d['image/png']))
        if o.get('output_type')=='error': print('ERROR cell', i, o['ename'], o['evalue'])
print('ok', name, 'images', k, 'size KB', os.path.getsize(f'{out}/{name}.ipynb')//1024)
