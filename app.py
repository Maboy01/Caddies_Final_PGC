#!/usr/bin/env python3
"""
app.py — Aplicación web del Club de Golf Serrezuela.

Construida con Streamlit, ofrece dos funcionalidades principales:
    1. Sistema de reservas de caddies: los socios pueden reservar caddies
       por categoría (1ra, 2da, 3ra) o elegir uno específico, con un
       sistema de anticipo del 50% y ventana de cancelación de 8 horas.
    2. Análisis de swing con IA: el socio sube un video de su swing y el
       modelo CNN+LSTM predice qué tipo de palo (wood/iron) se adapta
       mejor a su técnica.

Para ejecutar:
    streamlit run app.py
"""
from __future__ import annotations

import random
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

# Agrega la carpeta del proyecto al path para importar módulos locales
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.video import load_video_sequence
from model.classifier import CnnLstmClassifier

# ── Constantes ─────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_results" / "cnn_lstm" / "cnn_lstm_model.pt"

# Precios por ronda según categoría del caddie
PRECIOS = {"1ra": 90_000, "2da": 70_000, "3ra": 60_000}

# Lista inicial de caddies con sus datos (se copia al estado de sesión al iniciar)
CADDIES_INICIALES = [
    {"id": 1, "nombre": "Carlos Gómez",     "categoria": "1ra", "experiencia": "5 años",   "calificacion": 4.8, "disponible": True,  "rondas": 312},
    {"id": 2, "nombre": "Luis Martínez",    "categoria": "1ra", "experiencia": "4 años",   "calificacion": 4.6, "disponible": True,  "rondas": 241},
    {"id": 3, "nombre": "Felipe Vargas",    "categoria": "1ra", "experiencia": "3 años",   "calificacion": 4.5, "disponible": False, "rondas": 198},
    {"id": 4, "nombre": "Juan Pérez",       "categoria": "2da", "experiencia": "2 años",   "calificacion": 4.2, "disponible": True,  "rondas": 134},
    {"id": 5, "nombre": "Andrés Torres",    "categoria": "2da", "experiencia": "1.5 años", "calificacion": 4.0, "disponible": True,  "rondas": 89},
    {"id": 6, "nombre": "Diego Salcedo",    "categoria": "2da", "experiencia": "1 año",    "calificacion": 3.9, "disponible": False, "rondas": 62},
    {"id": 7, "nombre": "Miguel Rodríguez", "categoria": "3ra", "experiencia": "8 meses",  "calificacion": 3.8, "disponible": True,  "rondas": 41},
    {"id": 8, "nombre": "Santiago López",   "categoria": "3ra", "experiencia": "4 meses",  "calificacion": 3.6, "disponible": True,  "rondas": 28},
    {"id": 9, "nombre": "Camilo Ríos",      "categoria": "3ra", "experiencia": "2 meses",  "calificacion": 3.5, "disponible": True,  "rondas": 12},
]


# Etiquetas de categoría con emoji para mostrar en la UI
BADGE = {"1ra": "🥇 1ra Categoría", "2da": "🥈 2da Categoría", "3ra": "🥉 3ra Categoría"}

ESTADO_ICONO = {"activa": "🟢", "cancelada": "🔴", "completada": "🔵"}

# Credenciales de usuarios del sistema (solo socios del club)
USUARIOS = {
    "cesar":   {"nombre": "Cesar",   "password": "cesar123",   "rol": "Socio"},
    "karol":   {"nombre": "Karol",   "password": "karol123",   "rol": "Socio"},
    "michael": {"nombre": "Michael", "password": "michael123", "rol": "Socio"},
    "esteban": {"nombre": "Esteban", "password": "esteban123", "rol": "Socio"},
}

# Descripción de cada tipo de palo para mostrar al usuario tras el análisis de swing
CLUB_INFO = {
    "wood": ("🪵 Wood", "Palo largo — ideal para drives y fairway. Tu swing tiene el arco y la potencia para maximizar distancia."),
    "iron": ("⛳ Iron",  "Hierro — ideal para aproximaciones. Tu swing es preciso y controlado, perfecto para distancias medias."),
}

# Distancias estándar para jugadores nuevos (yardas)
DISTANCIAS_ESTANDAR: dict[str, int] = {
    "Driver":    245,
    "Madera 3":  215, "Madera 5": 195, "Madera 7": 178,
    "Híbrido 2": 198, "Híbrido 3": 185, "Híbrido 4": 172, "Híbrido 5": 160,
    "Hierro 3":  182, "Hierro 4": 170, "Hierro 5": 158,
    "Hierro 6":  145, "Hierro 7": 132, "Hierro 8": 118, "Hierro 9": 104,
    "PW": 90, "GW": 75, "SW": 60, "LW": 45,
}

# Categoría de cada palo según el modelo (wedge = fuera del alcance del modelo)
TIPO_PALO: dict[str, str] = {
    "Driver":    "wood",
    "Madera 3":  "wood", "Madera 5": "wood", "Madera 7": "wood",
    "Híbrido 2": "iron", "Híbrido 3": "iron", "Híbrido 4": "iron", "Híbrido 5": "iron",
    "Hierro 3":  "iron", "Hierro 4": "iron", "Hierro 5": "iron",
    "Hierro 6":  "iron", "Hierro 7": "iron", "Hierro 8": "iron", "Hierro 9": "iron",
    "PW": "wedge", "GW": "wedge", "SW": "wedge", "LW": "wedge",
}


# ── Modelo ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando modelo de IA...")
def cargar_modelo():
    """
    Carga el modelo CNN+LSTM desde el archivo checkpoint (.pt) guardado
    por train_model.py.

    El decorador @st.cache_resource hace que el modelo se cargue una sola
    vez y quede en memoria para todas las peticiones subsecuentes, evitando
    recargar los pesos en cada análisis de swing.

    Retorna una tupla (model, classes, seq_len, frame_size).
    Si el archivo no existe, retorna (None, None, 24, 112) para que el
    llamador pueda mostrar un mensaje de error apropiado.
    """
    if not MODEL_PATH.exists():
        return None, None, 24, 112  # Modelo no entrenado todavía

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    classes    = checkpoint["classes"]                   # ['iron', 'wood']
    seq_len    = checkpoint.get("sequence_length", 24)   # Frames por secuencia
    frame_size = checkpoint.get("frame_size", 112)       # Tamaño de cada frame

    model = CnnLstmClassifier(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Modo inferencia: desactiva dropout y BN de entrenamiento
    return model, classes, seq_len, frame_size


def predecir_swing(video_bytes: bytes) -> tuple[str, float, dict]:
    """
    Recibe un video en bytes, lo procesa con el modelo CNN+LSTM y predice
    qué tipo de palo se adapta mejor al swing del jugador.

    Pasos:
        1. Guarda el video en un archivo temporal (necesario para OpenCV)
        2. Extrae la secuencia de frames con load_video_sequence
        3. Pasa los frames por el modelo para obtener logits
        4. Aplica softmax para convertir logits a probabilidades
        5. Retorna la clase con mayor probabilidad, su confianza y
           el diccionario completo de probabilidades por clase

    Args:
        video_bytes: Contenido del archivo de video en bytes

    Retorna (clase_predicha, confianza, {clase: probabilidad}).
    En caso de error, retorna ("error", 0.0, {}).
    """
    model, classes, seq_len, frame_size = cargar_modelo()
    if model is None:
        return "error", 0.0, {}

    # Crea archivo temporal porque OpenCV necesita una ruta en disco
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        tmp = Path(f.name)

    try:
        # Extrae frames del video (sin eventos: usa todo el video)
        frames = load_video_sequence(tmp, np.array([], dtype=np.int32), seq_len, frame_size)

        tensor = torch.from_numpy(frames).unsqueeze(0)  # Agrega dimensión de batch: (1, T, C, H, W)

        with torch.no_grad():  # Sin gradientes: solo inferencia
            probs = F.softmax(model(tensor), dim=1).squeeze().numpy()

        idx        = int(np.argmax(probs))       # Índice de la clase con mayor probabilidad
        pred_class = classes[idx]                # Nombre de esa clase ('wood' o 'iron')
        confianza  = float(probs[idx])           # Probabilidad de la clase ganadora
        todas      = {cls: float(p) for cls, p in zip(classes, probs)}  # Todas las probs
        return pred_class, confianza, todas
    finally:
        tmp.unlink(missing_ok=True)  # Limpia el archivo temporal siempre, incluso si hay error


# ── Lógica de recomendación ────────────────────────────────────────────────────

def recomendar_por_yardas(yardas: int, bolsa: dict) -> tuple[str, int]:
    """Devuelve el palo de la bolsa más cercano a la distancia pedida."""
    palos = bolsa["palos"]
    if not palos:
        return "", 0
    nombre, distancia = min(palos.items(), key=lambda x: abs(x[1] - yardas))
    return nombre, distancia


def recomendar_combinado(
    yardas: int, bolsa: dict, clase_modelo: str, confianza: float
) -> dict:
    """Cruza la recomendación por yardas con el resultado del modelo de video."""
    palo_yardas, dist_yardas = recomendar_por_yardas(yardas, bolsa)
    tipo_yardas = TIPO_PALO.get(palo_yardas, "wedge")

    palos = bolsa["palos"]

    # Si el modelo tiene suficiente confianza, busca el mejor palo de ese tipo
    palos_modelo = {k: v for k, v in palos.items() if TIPO_PALO.get(k) == clase_modelo}
    if palos_modelo and confianza >= 0.60:
        palo_combinado, dist_combinado = min(
            palos_modelo.items(), key=lambda x: abs(x[1] - yardas)
        )
    else:
        palo_combinado, dist_combinado = palo_yardas, dist_yardas

    return {
        "palo_yardas":    palo_yardas,
        "dist_yardas":    dist_yardas,
        "tipo_yardas":    tipo_yardas,
        "palo_combinado": palo_combinado,
        "dist_combinado": dist_combinado,
        "coincide":       tipo_yardas == clase_modelo,
    }


# ── Estado ─────────────────────────────────────────────────────────────────────

def init_state() -> None:
    """
    Inicializa las variables del estado de sesión de Streamlit si no existen.

    st.session_state persiste entre re-renders de la página (Streamlit
    re-ejecuta todo el script en cada interacción del usuario).

    Variables inicializadas:
        usuario          : Usuario autenticado (None = no logueado)
        caddies          : Lista mutable de caddies con su disponibilidad actual
        reservas         : Lista de reservas realizadas en la sesión
        page             : Página activa ('inicio', 'reservar', 'mis_reservas', 'swing')
        caddie_pendiente : Caddie seleccionado esperando confirmación de reserva
    """
    if "usuario"          not in st.session_state:
        st.session_state.usuario          = None
    if "bolsa"            not in st.session_state:
        st.session_state.bolsa            = None

    if "caddies"          not in st.session_state:
        st.session_state.caddies          = [c.copy() for c in CADDIES_INICIALES]
    if "reservas"         not in st.session_state:
        st.session_state.reservas         = []
    if "page"             not in st.session_state:
        st.session_state.page             = "inicio"
    if "caddie_pendiente" not in st.session_state:
        st.session_state.caddie_pendiente = None


# ── Helpers ─────────────────────────────────────────────────────────────────────

def ir_a(page: str) -> None:
    """
    Cambia la página activa de la aplicación y fuerza un re-render de Streamlit.

    Args:
        page: Identificador de la página destino
              ('inicio', 'reservar', 'mis_reservas', 'swing')
    """
    st.session_state.page = page
    st.rerun()


def estrellas(rating: float) -> str:
    """
    Convierte una calificación numérica (1.0 – 5.0) a representación visual
    con símbolos de estrella.

    Ejemplos:
        4.8 → "★★★★★"   (4 llenas + redondea a completa)
        4.2 → "★★★★☆"
        3.6 → "★★★½☆"   (3 llenas + media estrella + 1 vacía)

    Args:
        rating: Calificación entre 0 y 5

    Retorna una cadena con caracteres ★ (llena), ½ (media) y ☆ (vacía).
    """
    llenas = int(rating)
    media  = 1 if (rating - llenas) >= 0.5 else 0
    vacias = 5 - llenas - media
    return "★" * llenas + ("½" if media else "") + "☆" * vacias


def cop(valor: int) -> str:
    """
    Formatea un valor entero como moneda colombiana (COP).

    Ejemplo: cop(90000) → "$90.000"

    Args:
        valor: Monto en pesos colombianos (entero)

    Retorna una cadena con formato "$X.XXX" usando puntos como separador de miles.
    """
    return f"${valor:,}".replace(",", ".")


# ── Páginas ────────────────────────────────────────────────────────────────────

def pagina_login() -> None:
    """
    Muestra el formulario de autenticación del sistema.

    Valida usuario y contraseña contra el diccionario USUARIOS.
    Si las credenciales son correctas, guarda el usuario en session_state
    y redirige a la página de inicio mediante st.rerun().
    Si son incorrectas, muestra un mensaje de error.

    El formulario se centra visualmente usando tres columnas de Streamlit.
    """
    _, col_centro, _ = st.columns([1, 1.2, 1])
    with col_centro:

        st.markdown("## ⛳ Club Serrezuela")
        st.markdown("### Iniciar sesión")
        st.markdown("---")
        with st.form("form_login"):
            usuario_input  = st.text_input("Usuario")
            password_input = st.text_input("Contraseña", type="password")
            submitted = st.form_submit_button("Entrar", use_container_width=True, type="primary")
        if submitted:
            key  = usuario_input.strip().lower()
            user = USUARIOS.get(key)
            if user and password_input == user["password"]:
                st.session_state.usuario = user
                st.rerun()  # Re-renderiza para mostrar la app principal
            else:
                st.error("Usuario o contraseña incorrectos.")


def pagina_bolsa() -> None:
    st.markdown("## 🏌️ Configura tu bolsa de palos")
    st.write("Solo necesitas hacerlo una vez. Usaremos esta información para recomendarte el palo correcto.")
    st.markdown("---")

    es_nuevo = st.toggle("Soy nuevo jugando golf", value=False)
    if es_nuevo:
        st.info("Usaremos distancias estándar — no necesitas ingresar nada más.")

    st.markdown("### ¿Qué palos tienes en tu bolsa?")

    col1, col2 = st.columns(2)
    with col1:
        tiene_driver   = st.checkbox("Driver", value=True)
        maderas_sel    = st.multiselect("Maderas de fairway", ["3", "5", "7"], default=["3", "5"])
        hibridos_sel   = st.multiselect("Híbridos", ["2", "3", "4", "5"], default=[])
    with col2:
        hierros_sel    = st.multiselect("Hierros", ["3", "4", "5", "6", "7", "8", "9"], default=["6", "7", "8", "9"])
        wedges_sel     = st.multiselect("Wedges", ["PW", "GW", "SW", "LW"], default=["PW"])

    palos_lista: list[str] = []
    if tiene_driver:
        palos_lista.append("Driver")
    palos_lista += [f"Madera {n}" for n in maderas_sel]
    palos_lista += [f"Híbrido {n}" for n in hibridos_sel]
    palos_lista += [f"Hierro {n}" for n in hierros_sel]
    palos_lista += wedges_sel

    distancias: dict[str, int] = {}

    if not es_nuevo and palos_lista:
        st.markdown("---")
        st.markdown("### ¿A cuántas yardas le pegas a cada palo?")
        st.caption("Usa tu distancia promedio en condiciones normales.")
        cols = st.columns(3)
        for i, palo in enumerate(palos_lista):
            with cols[i % 3]:
                distancias[palo] = st.number_input(
                    palo,
                    min_value=10, max_value=400,
                    value=DISTANCIAS_ESTANDAR.get(palo, 100),
                    step=5,
                    key=f"dist_{palo}",
                )
    else:
        distancias = {p: DISTANCIAS_ESTANDAR[p] for p in palos_lista if p in DISTANCIAS_ESTANDAR}

    st.markdown("---")
    if st.button("Guardar bolsa y continuar", type="primary", use_container_width=True):
        if not palos_lista:
            st.error("Selecciona al menos un palo.")
        else:
            st.session_state.bolsa = {"nuevo": es_nuevo, "palos": distancias}
            st.rerun()


def pagina_inicio() -> None:
    """
    Página de bienvenida con métricas rápidas y accesos directos.

    Muestra:
        - Número de caddies disponibles en este momento
        - Número de reservas activas del usuario
        - Nombre del socio logueado
        - Botones para ir a Reservar o a Analizar Swing
        - Tarjetas informativas con precios por categoría de caddie
    """
    st.markdown("## Bienvenido al Club Serrezuela")
    st.markdown("---")

    disponibles  = sum(1 for c in st.session_state.caddies if c["disponible"])
    reservas_act = sum(1 for r in st.session_state.reservas if r["estado"] == "activa")

    c1, c2, c3 = st.columns(3)
    c1.metric("Caddies disponibles", disponibles)
    c2.metric("Mis reservas activas", reservas_act)
    c3.metric("Socio", st.session_state.usuario["nombre"])

    st.markdown("---")
    st.markdown("### ¿Qué quieres hacer hoy?")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### 🏌️ Reservar un Caddie")
            st.write("Elige tu caddie favorito o solicita uno aleatorio por categoría.")
            if st.button("Reservar ahora", use_container_width=True, type="primary"):
                ir_a("reservar")
    with col2:
        with st.container(border=True):
            st.markdown("#### 🎯 Analiza tu Swing")
            st.write("Sube un video de tu swing y recibe recomendaciones con IA.")
            if st.button("Analizar swing", use_container_width=True):
                ir_a("swing")

    st.markdown("---")
    st.markdown("### Categorías de caddies")
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("#### 🥉 3ra Categoría")
            st.write("Sin experiencia mínima requerida")
            st.markdown(f"**{cop(60_000)}** por ronda")
    with col2:
        with st.container(border=True):
            st.markdown("#### 🥈 2da Categoría")
            st.write("1-2 años de experiencia")
            st.markdown(f"**{cop(70_000)}** por ronda")
    with col3:
        with st.container(border=True):
            st.markdown("#### 🥇 1ra Categoría")
            st.write("3 o más años de experiencia")
            st.markdown(f"**{cop(90_000)}** por ronda")


def _tarjeta_caddie(caddie: dict, boton: bool = True) -> None:
    """
    Renderiza la tarjeta de información de un caddie individual.

    Muestra:
        - Nombre, categoría con badge
        - Calificación en estrellas y número
        - Años de experiencia y rondas realizadas
        - Precio total y anticipo (50%)
        - Estado de disponibilidad
        - Botón "Reservar" (activo si está disponible) o "No disponible" (deshabilitado)

    Args:
        caddie : Diccionario con los datos del caddie
        boton  : Si True muestra el botón de reserva (False para vistas de solo lectura)
    """
    precio = PRECIOS[caddie["categoria"]]
    with st.container(border=True):
        col_info, col_btn = st.columns([4, 1])
        with col_info:
            st.markdown(f"**{caddie['nombre']}** — {BADGE[caddie['categoria']]}")
            st.markdown(
                f"{estrellas(caddie['calificacion'])} `{caddie['calificacion']}`  "
                f"|  {caddie['experiencia']}  |  {caddie['rondas']} rondas"
            )
            estado = "✅ Disponible" if caddie["disponible"] else "❌ No disponible"
            st.markdown(f"💰 {cop(precio)} — Anticipo (50%): **{cop(precio // 2)}**  |  {estado}")
        with col_btn:
            if boton:
                if caddie["disponible"]:
                    # Al hacer clic guarda el caddie en session_state y re-renderiza
                    # para mostrar el flujo de confirmación
                    if st.button("Reservar", key=f"res_{caddie['id']}", type="primary"):
                        st.session_state.caddie_pendiente = caddie
                        st.rerun()
                else:
                    st.button("No disponible", key=f"nd_{caddie['id']}", disabled=True)


def _confirmar_reserva(caddie: dict) -> None:
    """
    Muestra el resumen de la reserva y los botones de confirmación/cancelación.

    Cuando el usuario confirma:
        - Crea un registro de reserva con ID aleatorio, fecha y límite de cancelación
        - Marca el caddie como no disponible en session_state
        - Agrega la reserva a la lista de reservas del usuario
        - Limpia caddie_pendiente para volver al flujo normal

    La ventana de cancelación con reembolso es de 8 horas desde la confirmación.

    Args:
        caddie: Diccionario con los datos del caddie a reservar
    """
    precio   = PRECIOS[caddie["categoria"]]
    anticipo = precio // 2  # 50% de anticipo

    st.markdown("### Confirmar reserva")
    with st.container(border=True):
        st.markdown(f"**Caddie:** {caddie['nombre']}")
        st.markdown(f"**Categoría:** {BADGE[caddie['categoria']]}")
        st.markdown(f"**Calificación:** {estrellas(caddie['calificacion'])} {caddie['calificacion']}")
        st.markdown(f"**Precio total:** {cop(precio)}")
        st.markdown(f"**Anticipo a pagar ahora (50%):** {cop(anticipo)}")
        st.info("Tienes **8 horas** para cancelar y recuperar el anticipo completo.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Confirmar y pagar", type="primary", use_container_width=True):
            reserva = {
                "id":                 random.randint(1000, 9999),
                "caddie":             caddie.copy(),
                "precio_total":       precio,
                "anticipo":           anticipo,
                "fecha_reserva":      datetime.now(),
                "limite_cancelacion": datetime.now() + timedelta(hours=8),
                "estado":             "activa",
            }
            # Marca el caddie como no disponible para otros socios
            for c in st.session_state.caddies:
                if c["id"] == caddie["id"]:
                    c["disponible"] = False
                    break
            st.session_state.reservas.append(reserva)
            st.session_state.caddie_pendiente = None  # Limpia el pendiente
            st.success(
                f"Reserva #{reserva['id']} confirmada. "
                f"Se cobró el anticipo de {cop(anticipo)}. "
                f"{caddie['nombre']} ha sido notificado."
            )
            st.balloons()
            st.rerun()
    with col2:
        if st.button("Volver", use_container_width=True):
            st.session_state.caddie_pendiente = None
            st.rerun()


def pagina_reservar() -> None:
    """
    Página para seleccionar y reservar un caddie.

    Ofrece dos modos de elección:
        1. Elegir caddie específico: muestra la lista filtrable por categoría
           con tarjeta de cada caddie y botón de reserva individual.
        2. Caddie aleatorio por categoría: muestra tres botones (3ra, 2da, 1ra)
           que seleccionan aleatoriamente un caddie disponible de esa categoría.

    Si hay un caddie_pendiente en session_state, muestra el flujo de
    confirmación de reserva en lugar de la lista.
    """
    st.markdown("## 🏌️ Reservar Caddie")

    # Si ya se eligió un caddie, pasa directamente a la confirmación
    if st.session_state.caddie_pendiente:
        _confirmar_reserva(st.session_state.caddie_pendiente)
        return

    modo = st.radio(
        "¿Cómo quieres elegir?",
        ["Elegir caddie específico", "Caddie aleatorio por categoría"],
        horizontal=True,
    )
    st.markdown("---")

    if modo == "Elegir caddie específico":
        filtro = st.selectbox("Filtrar por categoría", ["Todas", "1ra", "2da", "3ra"])
        lista  = [c for c in st.session_state.caddies if filtro == "Todas" or c["categoria"] == filtro]
        for caddie in lista:
            _tarjeta_caddie(caddie)
    else:
        st.markdown("### Selecciona la categoría que deseas")
        col1, col2, col3 = st.columns(3)

        def solicitar_aleatorio(cat: str) -> None:
            """Elige aleatoriamente un caddie disponible de la categoría dada."""
            disponibles = [
                c for c in st.session_state.caddies
                if c["categoria"] == cat and c["disponible"]
            ]
            if not disponibles:
                st.error(f"No hay caddies de {cat} categoría disponibles ahora.")
                return
            st.session_state.caddie_pendiente = random.choice(disponibles)
            st.rerun()

        with col1:
            with st.container(border=True):
                st.markdown("#### 🥉 3ra Categoría")
                st.write("Sin experiencia mínima")
                st.markdown(f"**{cop(60_000)}** / ronda")
                if st.button("Solicitar 3ra", use_container_width=True):
                    solicitar_aleatorio("3ra")
        with col2:
            with st.container(border=True):
                st.markdown("#### 🥈 2da Categoría")
                st.write("1-2 años de experiencia")
                st.markdown(f"**{cop(70_000)}** / ronda")
                if st.button("Solicitar 2da", use_container_width=True):
                    solicitar_aleatorio("2da")
        with col3:
            with st.container(border=True):
                st.markdown("#### 🥇 1ra Categoría")
                st.write("3+ años de experiencia")
                st.markdown(f"**{cop(90_000)}** / ronda")
                if st.button("Solicitar 1ra", use_container_width=True):
                    solicitar_aleatorio("1ra")


def pagina_mis_reservas() -> None:
    """
    Muestra todas las reservas del usuario en la sesión actual.

    Para cada reserva activa calcula el tiempo restante dentro de la
    ventana de cancelación con reembolso (8 horas desde la reserva):

        - Si está en tiempo: muestra botón de cancelación con reembolso total
          del anticipo y el tiempo restante.
        - Si el tiempo expiró: ofrece cancelación sin reembolso (el anticipo
          se transfiere al caddie).

    Al cancelar:
        - Cambia el estado de la reserva a "cancelada"
        - Libera al caddie (disponible=True) para otros socios
    """
    st.markdown("## 📋 Mis Reservas")

    if not st.session_state.reservas:
        st.info("Aún no tienes reservas.")
        if st.button("Reservar un caddie", type="primary"):
            ir_a("reservar")
        return

    for i, reserva in enumerate(st.session_state.reservas):
        caddie    = reserva["caddie"]
        ahora     = datetime.now()
        activa    = reserva["estado"] == "activa"
        en_tiempo = ahora < reserva["limite_cancelacion"]

        with st.container(border=True):
            icono = ESTADO_ICONO.get(reserva["estado"], "⚪")
            st.markdown(f"**Reserva #{reserva['id']}** — {icono} {reserva['estado'].capitalize()}")
            st.markdown(f"**Caddie:** {caddie['nombre']} — {BADGE[caddie['categoria']]}")
            st.markdown(
                f"**Anticipo pagado:** {cop(reserva['anticipo'])}  |  "
                f"**Saldo pendiente:** {cop(reserva['precio_total'] - reserva['anticipo'])}"
            )
            st.markdown(f"**Reservado el:** {reserva['fecha_reserva'].strftime('%d/%m/%Y %H:%M')}")

            if activa:
                if en_tiempo:
                    # Calcula y muestra el tiempo restante para cancelar con reembolso
                    restante = reserva["limite_cancelacion"] - ahora
                    h = int(restante.total_seconds() // 3600)
                    m = int((restante.total_seconds() % 3600) // 60)
                    st.warning(f"Puedes cancelar con reembolso durante {h}h {m}m más.")
                    if st.button(f"Cancelar reserva #{reserva['id']}", key=f"cancel_{i}"):
                        reserva["estado"] = "cancelada"
                        # Libera al caddie para que pueda ser reservado de nuevo
                        for c in st.session_state.caddies:
                            if c["id"] == caddie["id"]:
                                c["disponible"] = True
                                break
                        st.success(f"Reserva cancelada. Se reembolsarán {cop(reserva['anticipo'])}.")
                        st.rerun()
                else:
                    # Fuera de la ventana: anticipo ya no se reembolsa
                    st.error("Ya no puedes cancelar con reembolso. El anticipo será transferido al caddie si no te presentas.")
                    if st.button(f"Cancelar sin reembolso #{reserva['id']}", key=f"cancel_nr_{i}"):
                        reserva["estado"] = "cancelada"
                        for c in st.session_state.caddies:
                            if c["id"] == caddie["id"]:
                                c["disponible"] = True
                                break
                        st.info(
                            f"Reserva cancelada. El anticipo de {cop(reserva['anticipo'])} "
                            f"fue transferido a {caddie['nombre']}."
                        )
                        st.rerun()


def pagina_swing() -> None:
    """
    Página de análisis de swing con inteligencia artificial.

    Flujo:
        1. Verifica que el modelo esté entrenado y disponible.
        2. Permite al usuario subir un video de su swing (mp4/avi/mov/mkv).
        3. Muestra consejos de grabación mientras no hay video cargado.
        4. Al presionar "Analizar swing", llama a predecir_swing() y muestra:
            - Palo recomendado (wood o iron) y nivel de confianza
            - Barra de progreso con la distribución de probabilidades
            - Mensaje de recomendación según el nivel de confianza:
                ≥75%: mensaje positivo (alta certeza)
                55-75%: advertencia (certeza media)
                <55%: sugerencia de mejorar el video
    """
    st.markdown("## 🎯 Analiza tu Swing")
    st.markdown("---")

    bolsa = st.session_state.bolsa

    # ── Sección 1: recomendación por yardas ───────────────────────────────────
    st.markdown("### 📍 ¿A cuántas yardas estás de la bandera?")
    col_yard, col_btn = st.columns([2, 1])
    with col_yard:
        yardas = st.number_input(
            "Yardas a la bandera",
            min_value=1, max_value=600, value=150, step=5,
            label_visibility="collapsed",
        )
    with col_btn:
        buscar_yardas = st.button("Recomendar palo", use_container_width=True, type="primary")

    if buscar_yardas:
        palo, distancia = recomendar_por_yardas(yardas, bolsa)
        tipo = TIPO_PALO.get(palo, "wedge")
        diferencia = abs(distancia - yardas)

        with st.container(border=True):
            st.markdown(f"#### Recomendación: **{palo}**")
            st.markdown(f"Tu distancia con este palo: **{distancia} yds** — diferencia: {diferencia} yds")

            if tipo == "wedge":
                st.info("Para esta distancia corta te recomendamos un **wedge**. El análisis de video no aplica para wedges.")
            elif diferencia <= 10:
                st.success(f"Distancia ideal para el **{palo}**.")
            elif diferencia <= 20:
                st.warning(f"Es la mejor opción de tu bolsa, aunque estás a {diferencia} yds de tu distancia ideal.")
            else:
                st.warning(f"No tienes un palo con distancia exacta para {yardas} yds. El **{palo}** es el más cercano.")

    st.markdown("---")

    # ── Sección 2: análisis de video + recomendación combinada ───────────────
    st.markdown("### 🎥 Análisis de swing con IA")
    st.write("Sube un video de tu swing para cruzar el análisis con la distancia y obtener la mejor recomendación.")

    model, *_ = cargar_modelo()
    if model is None:
        st.error("Modelo no encontrado. Ejecuta `python train_model.py` primero.")
        return

    archivo = st.file_uploader(
        "Sube tu video de swing",
        type=["mp4", "avi", "mov", "mkv"],
        help="Incluye el swing completo. Recomendado: 5-30 segundos.",
    )

    if archivo is None:
        # Muestra guía de grabación cuando no hay video
        with st.container(border=True):
            st.markdown("**Consejos para un buen video:**")
            st.markdown(
                "- Graba desde un ángulo lateral (lado del caddie)\n"
                "- Buena iluminación, sin contraluz\n"
                "- Incluye el swing completo: backswing, impacto y follow-through\n"
                "- 5-15 segundos es ideal\n"
                "- ⚠️ El modelo no detecta wedges — para distancias cortas usa solo la recomendación por yardas"
            )
        return

    st.video(archivo)  # Muestra el video al usuario antes del análisis

    if st.button("Analizar swing", type="primary", use_container_width=True):
        with st.spinner("Analizando tu swing..."):
            clase, confianza, todas = predecir_swing(archivo.read())

        if clase == "error":
            st.error("No se pudo procesar el video. Intenta con otro archivo.")
            return

        st.markdown("---")
        st.markdown("### Resultado del modelo")

        nombre_club, descripcion = CLUB_INFO.get(clase, (clase, ""))
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tipo de swing detectado", nombre_club)
            st.metric("Confianza del modelo", f"{confianza * 100:.1f}%")
        with col2:
            st.markdown("**Distribución de probabilidades:**")
            # Ordena de mayor a menor probabilidad
            for cls, prob in sorted(todas.items(), key=lambda x: -x[1]):
                etiqueta = CLUB_INFO.get(cls, (cls,))[0]
                st.progress(prob, text=f"{etiqueta}: {prob * 100:.1f}%")

        # Recomendación combinada
        st.markdown("---")
        st.markdown("### Recomendación final")

        resultado = recomendar_combinado(yardas, bolsa, clase, confianza)
        palo_final = resultado["palo_combinado"]
        tipo_final = TIPO_PALO.get(palo_final, "wedge")

        with st.container(border=True):
            st.markdown(f"#### Usa: **{palo_final}**")
            st.markdown(f"Tu distancia con este palo: **{resultado['dist_combinado']} yds** — bandera a **{yardas} yds**")

            if resultado["coincide"]:
                st.success(
                    f"El modelo detectó un swing de **{nombre_club}** y tu bolsa confirma "
                    f"que el **{palo_final}** es el palo correcto para {yardas} yds."
                )
            elif confianza < 0.60:
                st.info(
                    f"El modelo no está muy seguro del tipo de swing ({confianza*100:.0f}% confianza). "
                    f"La recomendación se basa principalmente en la distancia: **{palo_final}**."
                )
            else:
                tipo_modelo_str = "madera/driver" if clase == "wood" else "hierro"
                st.warning(
                    f"El modelo sugiere un swing de **{tipo_modelo_str}**, pero para {yardas} yds "
                    f"el palo más adecuado de tu bolsa es el **{palo_final}**. "
                    f"Considera ajustar tu técnica o revisar tus distancias."
                )

        if confianza < 0.55:
            st.info("Intenta con un video con mejor iluminación y ángulo lateral claro para mejorar la precisión del modelo.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Punto de entrada de la aplicación Streamlit.

    Responsabilidades:
        1. Configura la página (título, ícono, layout ancho)
        2. Inicializa el estado de sesión
        3. Si no hay usuario logueado: muestra solo la página de login
        4. Si hay usuario logueado: muestra la barra lateral de navegación
           y renderiza la página activa según st.session_state.page

    Streamlit ejecuta esta función completa en cada interacción del usuario.
    """
    st.set_page_config(
        page_title="Club Serrezuela — Caddies",
        page_icon="⛳",
        layout="wide",
    )

    init_state()

    if st.session_state.usuario is None:
        pagina_login()
        return

    if st.session_state.bolsa is None:
        pagina_bolsa()
        return

    usuario = st.session_state.usuario

    with st.sidebar:
        st.markdown("## ⛳ Club Serrezuela")
        st.markdown("---")

        # Botones de navegación — cada clic actualiza session_state.page y recarga
        paginas = [
            ("🏠 Inicio",          "inicio"),
            ("🏌️ Reservar Caddie",  "reservar"),
            ("📋 Mis Reservas",     "mis_reservas"),
            ("🎯 Analiza tu Swing", "swing"),
        ]
        for label, page in paginas:
            if st.button(label, use_container_width=True, key=f"nav_{page}"):
                st.session_state.page = page
                st.rerun()

        st.markdown("---")
        st.markdown(f"**{usuario['rol']}:** {usuario['nombre']}")

        bolsa = st.session_state.bolsa
        n_palos = len(bolsa["palos"])
        tipo_jugador = "Nuevo jugador" if bolsa["nuevo"] else "Jugador con experiencia"
        st.caption(f"{tipo_jugador} · {n_palos} palos en bolsa")

        if st.button("Editar bolsa", use_container_width=True):
            st.session_state.bolsa = None
            st.rerun()

        if st.button("Cerrar sesión", use_container_width=True):
            st.session_state.usuario = None
            st.session_state.bolsa   = None
            st.session_state.page    = "inicio"
            st.rerun()

    page = st.session_state.page
    if   page == "inicio":       pagina_inicio()
    elif page == "reservar":     pagina_reservar()
    elif page == "mis_reservas": pagina_mis_reservas()
    elif page == "swing":        pagina_swing()


if __name__ == "__main__":
    main()
