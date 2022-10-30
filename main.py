import os
import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import predict
from dotenv import load_dotenv
load_dotenv('rutas_cfg.env')
PORT = os.getenv('PORT', 5000)
print(f'Puerto configurado: {PORT}')

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

@app.get("/")
def get_home(request: Request):
    pass
    
    # resultados.to_markdown('resultados.md')
    
    return templates.TemplateResponse('website.html', {"request": request})

@app.post("/upload/")
async def upload_file(file: UploadFile):
    # Persistir el archivo recibido
    ruta_archivo_entrada = 'uploads/' + file.filename.replace(' ', '')
    f = open(ruta_archivo_entrada, 'wb')
    contents = await file.read()
    f.write(contents)
    
    # Llamada a detect
    ruta_archivo_salida = predict.run(ruta_archivo_entrada)
    ruta_archivo_salida = os.path.abspath(ruta_archivo_salida)
    print('Salida: ' + ruta_archivo_salida)
    
    return FileResponse(ruta_archivo_salida)


    

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=PORT, log_level='info')

