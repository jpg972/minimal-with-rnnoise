import ctypes
import numpy as np
import os

# ------------------------------------------------------------------
# CONFIGURACIÓN DE RUTA A LA DLL
# ------------------------------------------------------------------
# Carpeta donde está este archivo 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construimos la ruta relativa hacia rnnoise.dll
dll_path = os.path.normpath(
    os.path.join(BASE_DIR, "..", "..", "rnnoise", "build", "Release", "rnnoise.dll")
)

# Cargamos la DLL (igual que antes, pero ahora con ruta relativa)
rnnoise = ctypes.cdll.LoadLibrary(dll_path)
# Definición del tipo de puntero RNNState (opaco)
class RNNState(ctypes.Structure):
    pass

# ------------------------------------------------------------------
# DECLARACIÓN DE PROTOTIPOS DE FUNCIONES 
# ------------------------------------------------------------------
rnnoise.rnnoise_create.restype = ctypes.POINTER(RNNState)
rnnoise.rnnoise_destroy.argtypes = [ctypes.POINTER(RNNState)]
rnnoise.rnnoise_process_frame.argtypes = [
    ctypes.POINTER(RNNState),
    ctypes.POINTER(ctypes.c_float), # out_buf (salida: audio limpio)
    ctypes.POINTER(ctypes.c_float)  # in_buf (entrada: audio sucio)
]
rnnoise.rnnoise_process_frame.restype = ctypes.c_float

# ------------------------------------------------------------------
# CLASE ENVOLTORIO
# ------------------------------------------------------------------
class RNNoise:
    FRAME_SIZE = 480  # 10 ms @ 48 kHz

    def __init__(self):
        self.state = rnnoise.rnnoise_create()
        if not self.state:
            raise RuntimeError("Error creando el estado RNNoise (DLL no inicializada)")

    def process_frame(self, frame: np.ndarray):
        """
        Procesa un frame de 480 muestras (float32).
        
        Args:
            frame: Array de NumPy (dtype=float32) de 480 muestras.
        
        Returns:
            Tuple[np.ndarray, float]: Frame limpio y probabilidad VAD.
        """
        if frame.dtype != np.float32 or len(frame) != self.FRAME_SIZE:
            raise ValueError(f"El frame debe ser np.float32 con {self.FRAME_SIZE} muestras.")

        # Conversión eficiente: obtenemos el puntero directo a la memoria del array
        # Este es el cambio clave para el rendimiento
        in_buf = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Array de salida de NumPy vacío (dtype=float32)
        out_array = np.zeros(self.FRAME_SIZE, dtype=np.float32)
        out_buf = out_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Ejecutar la función de la DLL
        vad_prob = rnnoise.rnnoise_process_frame(self.state, out_buf, in_buf)
        
        return out_array, vad_prob

    def destroy(self):
        if self.state:
            rnnoise.rnnoise_destroy(self.state)
            self.state = None
    
    # Aseguramos que la memoria se libere cuando el objeto es recolectado
    def __del__(self):
        self.destroy()

# ------------------------------------------------------------------
# PRUEBA DE EJECUCIÓN
# ------------------------------------------------------------------
if __name__ == '__main__':
    # Simular un frame de audio (ruido bajo)
    # Importante: el dtype debe ser np.float32
    fake_audio = np.random.uniform(-0.1, 0.1, RNNoise.FRAME_SIZE).astype(np.float32)

    try:
        denoiser = RNNoise()
        print("✅ Instancia de RNNoise creada con éxito. Procesando frame de prueba...")
        
        # Ejecutar el procesamiento
        clean_frame, vad = denoiser.process_frame(fake_audio)
        
        print("\n--- Resultados de la prueba ---")
        print(f"✅ Frame procesado correctamente.")
        print(f"   Probabilidad VAD (Voz): {vad:.4f}")
        print(f"   Tamaño del array de salida: {len(clean_frame)} muestras.")
        print("------------------------------\n")
        
        # La memoria se libera automáticamente al salir del bloque, pero llamamos a destroy
        denoiser.destroy() 
    except Exception as e:
        print(f"❌ FALLO CRÍTICO DE EJECUCIÓN: {e}")