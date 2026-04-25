#!/usr/bin/env python3
"""App de reserva de caddies y análisis de swing — Club Serrezuela."""
from __future__ import annotations

import os
import random
import sys
import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.video import load_video_sequence
from model.classifier import CnnLstmClassifier


@st.cache_resource
def get_supabase() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def parse_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=None) if dt.tzinfo else dt

# ── Constantes ─────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_results" / "cnn_lstm" / "cnn_lstm_model.pt"

PRECIOS = {"1ra": 90_000, "2da": 70_000, "3ra": 60_000}

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

BADGE = {"1ra": "🥇 1ra Categoría", "2da": "🥈 2da Categoría", "3ra": "🥉 3ra Categoría"}
ESTADO_ICONO = {"activa": "🟢", "cancelada": "🔴", "completada": "🔵"}

USUARIOS = {
    "cesar":    {"nombre": "Cesar",    "password": "cesar123",    "rol": "Socio"},
    "karol":    {"nombre": "Karol",    "password": "karol123",    "rol": "Socio"},
    "michael":  {"nombre": "Michael",  "password": "michael123",  "rol": "Socio"},
    "esteban":  {"nombre": "Esteban",  "password": "esteban123",  "rol": "Socio"},
}

CLUB_INFO = {
    "wood":     ("🪵 Wood",    "Palo largo — ideal para drives y fairway. Tu swing tiene el arco y la potencia para maximizar distancia."),
    "iron":     ("⛳ Iron",    "Hierro — ideal para aproximaciones. Tu swing es preciso y controlado, perfecto para distancias medias."),
    "no_golf":  ("❌ No Golf", "No se detectó un swing de golf en el video."),
}


# ── Modelo ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando modelo de IA...")
def cargar_modelo():
    if not MODEL_PATH.exists():
        return None, None, 24, 112
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    classes    = checkpoint["classes"]
    seq_len    = checkpoint.get("sequence_length", 24)
    frame_size = checkpoint.get("frame_size", 112)
    model = CnnLstmClassifier(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes, seq_len, frame_size


def predecir_swing(video_bytes: bytes) -> tuple[str, float, dict]:
    model, classes, seq_len, frame_size = cargar_modelo()
    if model is None:
        return "error", 0.0, {}

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        tmp = Path(f.name)

    try:
        frames = load_video_sequence(tmp, np.array([], dtype=np.int32), seq_len, frame_size)
        tensor = torch.from_numpy(frames).unsqueeze(0)   # (1, T, C, H, W)
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1).squeeze().numpy()
        idx        = int(np.argmax(probs))
        pred_class = classes[idx]
        confianza  = float(probs[idx])
        todas      = {cls: float(p) for cls, p in zip(classes, probs)}
        return pred_class, confianza, todas
    finally:
        tmp.unlink(missing_ok=True)


# ── Estado ─────────────────────────────────────────────────────────────────────

def init_state() -> None:
    if "usuario"          not in st.session_state:
        st.session_state.usuario          = None
    if "caddies"          not in st.session_state:
        result = get_supabase().table("caddies").select("*").execute()
        st.session_state.caddies          = result.data
    if "page"             not in st.session_state:
        st.session_state.page             = st.query_params.get("page", "inicio")
    if "caddie_pendiente" not in st.session_state:
        st.session_state.caddie_pendiente = None
    if "close_sidebar" not in st.session_state:
        st.session_state.close_sidebar = False


# ── Helpers ─────────────────────────────────────────────────────────────────────

def ir_a(page: str) -> None:
    st.session_state.page = page
    st.query_params["page"] = page
    st.rerun()


def estrellas(rating: float) -> str:
    llenas = int(rating)
    media  = 1 if (rating - llenas) >= 0.5 else 0
    vacias = 5 - llenas - media
    return "★" * llenas + ("½" if media else "") + "☆" * vacias


def cop(valor: int) -> str:
    return f"\\${valor:,}".replace(",", ".")


# ── Páginas ────────────────────────────────────────────────────────────────────

def pagina_login() -> None:
    col_izq, col_centro, col_der = st.columns([1, 1.2, 1])
    with col_centro:
        st.markdown("## ⛳ Club Serrezuela")
        st.markdown("### Iniciar sesión")
        st.markdown("---")

        with st.form("form_login"):
            usuario_input = st.text_input("Usuario")
            password_input = st.text_input("Contraseña", type="password")
            submitted = st.form_submit_button("Entrar", use_container_width=True, type="primary")

        if submitted:
            key = usuario_input.strip().lower()
            result = get_supabase().table("usuarios").select("*").eq("username", key).eq("password", password_input).execute()
            if result.data:
                st.session_state.usuario = result.data[0]
                session = get_supabase().table("sessions").insert({"username": key}).execute()
                st.query_params["token"] = session.data[0]["token"]
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")


def pagina_inicio() -> None:
    st.markdown("## Bienvenido al Club Serrezuela")
    st.markdown("---")

    disponibles  = sum(1 for c in st.session_state.caddies if c["disponible"])
    res_result   = get_supabase().table("reservas").select("id").eq("usuario_username", st.session_state.usuario["username"]).eq("estado", "activa").execute()
    reservas_act = len(res_result.data)

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
            st.markdown(
                f"💰 {cop(precio)} — Anticipo (50%): **{cop(precio // 2)}**  |  {estado}"
            )
        with col_btn:
            if boton:
                if caddie["disponible"]:
                    if st.button("Reservar", key=f"res_{caddie['id']}", type="primary"):
                        st.session_state.caddie_pendiente = caddie
                        st.rerun()
                else:
                    st.button("No disponible", key=f"nd_{caddie['id']}", disabled=True)


def _confirmar_reserva(caddie: dict) -> None:
    precio   = PRECIOS[caddie["categoria"]]
    anticipo = precio // 2

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
            sb = get_supabase()
            reserva_id = random.randint(1000, 9999)
            fecha      = datetime.now()
            limite     = fecha + timedelta(hours=8)
            sb.table("reservas").insert({
                "id":                 reserva_id,
                "usuario_username":   st.session_state.usuario["username"],
                "caddie_id":          caddie["id"],
                "precio_total":       precio,
                "anticipo":           anticipo,
                "fecha_reserva":      fecha.isoformat(),
                "limite_cancelacion": limite.isoformat(),
                "estado":             "activa",
            }).execute()
            sb.table("caddies").update({"disponible": False}).eq("id", caddie["id"]).execute()
            for c in st.session_state.caddies:
                if c["id"] == caddie["id"]:
                    c["disponible"] = False
                    break
            st.session_state.caddie_pendiente = None
            st.success(
                f"Reserva #{reserva_id} confirmada. "
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
    st.markdown("## 🏌️ Reservar Caddie")

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
        lista = [
            c for c in st.session_state.caddies
            if filtro == "Todas" or c["categoria"] == filtro
        ]
        for caddie in lista:
            _tarjeta_caddie(caddie)

    else:
        st.markdown("### Selecciona la categoría que deseas")
        col1, col2, col3 = st.columns(3)

        def solicitar_aleatorio(cat: str) -> None:
            disponibles = [c for c in st.session_state.caddies if c["categoria"] == cat and c["disponible"]]
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
    st.markdown("## 📋 Mis Reservas")

    sb       = get_supabase()
    username = st.session_state.usuario["username"]
    result   = sb.table("reservas").select("*, caddies(*)").eq("usuario_username", username).neq("estado", "cancelada").order("fecha_reserva", desc=True).execute()
    reservas = result.data

    if not reservas:
        st.info("Aún no tienes reservas.")
        if st.button("Reservar un caddie", type="primary"):
            ir_a("reservar")
        return

    for i, reserva in enumerate(reservas):
        caddie    = reserva["caddies"]
        ahora     = datetime.now()
        activa    = reserva["estado"] == "activa"
        en_tiempo = ahora < parse_dt(reserva["limite_cancelacion"])

        with st.container(border=True):
            icono = ESTADO_ICONO.get(reserva["estado"], "⚪")
            st.markdown(f"**Reserva #{reserva['id']}** — {icono} {reserva['estado'].capitalize()}")
            st.markdown(f"**Caddie:** {caddie['nombre']} — {BADGE[caddie['categoria']]}")
            st.markdown(
                f"**Anticipo pagado:** {cop(reserva['anticipo'])}  |  "
                f"**Saldo pendiente:** {cop(reserva['precio_total'] - reserva['anticipo'])}"
            )
            st.markdown(f"**Reservado el:** {parse_dt(reserva['fecha_reserva']).strftime('%d/%m/%Y %H:%M')}")

            if activa:
                if en_tiempo:
                    restante = parse_dt(reserva["limite_cancelacion"]) - ahora
                    h = int(restante.total_seconds() // 3600)
                    m = int((restante.total_seconds() % 3600) // 60)
                    st.warning(f"Puedes cancelar con reembolso durante {h}h {m}m más.")
                    if st.button(f"Cancelar reserva #{reserva['id']}", key=f"cancel_{i}"):
                        sb.table("reservas").update({"estado": "cancelada"}).eq("id", reserva["id"]).execute()
                        sb.table("caddies").update({"disponible": True}).eq("id", caddie["id"]).execute()
                        for c in st.session_state.caddies:
                            if c["id"] == caddie["id"]:
                                c["disponible"] = True
                                break
                        st.success(f"Reserva cancelada. Se reembolsarán {cop(reserva['anticipo'])}.")
                        st.rerun()
                else:
                    st.error("Ya no puedes cancelar con reembolso. El anticipo será transferido al caddie si no te presentas.")
                    if st.button(f"Cancelar sin reembolso #{reserva['id']}", key=f"cancel_nr_{i}"):
                        sb.table("reservas").update({"estado": "cancelada"}).eq("id", reserva["id"]).execute()
                        sb.table("caddies").update({"disponible": True}).eq("id", caddie["id"]).execute()
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
    st.markdown("## 🎯 Analiza tu Swing")
    st.write(
        "Sube un video de tu swing y nuestro modelo de IA te recomendará qué tipo de palo "
        "se adapta mejor a tu técnica."
    )

    model, *_ = cargar_modelo()
    if model is None:
        st.error(
            f"Modelo no encontrado en `{MODEL_PATH.relative_to(BASE_DIR)}`.\n\n"
            "Ejecuta `python train_model.py` para entrenar el modelo primero."
        )
        return

    st.markdown("---")

    archivo = st.file_uploader(
        "Sube tu video de swing",
        type=["mp4", "avi", "mov", "mkv"],
        help="Incluye el swing completo. Recomendado: 5-30 segundos.",
    )

    if archivo is None:
        with st.container(border=True):
            st.markdown("**Consejos para un buen análisis:**")
            st.markdown(
                "- Graba desde un ángulo lateral (lado del caddie)\n"
                "- Buena iluminación, sin contraluz\n"
                "- Incluye el swing completo: backswing, impacto y follow-through\n"
                "- Video de 5-15 segundos es ideal"
            )
        return

    st.video(archivo)

    if st.button("Analizar swing", type="primary", use_container_width=True):
        with st.spinner("Analizando tu swing..."):
            clase, confianza, todas = predecir_swing(archivo.read())

        if clase == "error":
            st.error("No se pudo procesar el video. Intenta con otro archivo.")
            return

        st.markdown("---")
        st.markdown("### Resultado")

        nombre_club, descripcion = CLUB_INFO.get(clase, (clase, ""))

        if clase == "no_golf":
            st.warning(
                f"**No se detectó un swing de golf en el video** "
                f"(confianza: {confianza * 100:.1f}%). "
                "Asegúrate de subir un video con un swing completo y ángulo lateral claro."
            )
            st.markdown("**Distribución de probabilidades:**")
            for cls, prob in sorted(todas.items(), key=lambda x: -x[1]):
                etiqueta = CLUB_INFO.get(cls, (cls,))[0]
                st.progress(prob, text=f"{etiqueta}: {prob * 100:.1f}%")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Palo recomendado", nombre_club)
            st.metric("Confianza", f"{confianza * 100:.1f}%")
        with col2:
            st.markdown("**Distribución de probabilidades:**")
            for cls, prob in sorted(todas.items(), key=lambda x: -x[1]):
                etiqueta = CLUB_INFO.get(cls, (cls,))[0]
                st.progress(prob, text=f"{etiqueta}: {prob * 100:.1f}%")

        st.markdown("---")
        st.success(f"**{nombre_club}** — {descripcion}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Club Serrezuela — Caddies",
        page_icon="⛳",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    init_state()

    if st.session_state.usuario is None:
        token = st.query_params.get("token")
        if token:
            session = get_supabase().table("sessions").select("username").eq("token", token).execute()
            if session.data:
                username = session.data[0]["username"]
                result = get_supabase().table("usuarios").select("*").eq("username", username).execute()
                if result.data:
                    st.session_state.usuario = result.data[0]
                    st.rerun()

    if st.session_state.usuario is None:
        pagina_login()
        return

    usuario = st.session_state.usuario

    # Sidebar
    with st.sidebar:
        st.markdown("## ⛳ Club Serrezuela")
        st.markdown("---")

        paginas = [
            ("🏠 Inicio",          "inicio"),
            ("🏌️ Reservar Caddie",  "reservar"),
            ("📋 Mis Reservas",     "mis_reservas"),
            ("🎯 Analiza tu Swing", "swing"),
        ]
        for label, page in paginas:
            if st.button(label, use_container_width=True, key=f"nav_{page}"):
                st.session_state.page = page
                st.query_params["page"] = page
                st.session_state.close_sidebar = True
                st.rerun()

        st.markdown("---")
        st.markdown(f"**{usuario['rol']}:** {usuario['nombre']}")
        st.caption("Club Serrezuela")
        if st.button("Cerrar sesión", use_container_width=True):
            token = st.query_params.get("token")
            if token:
                get_supabase().table("sessions").delete().eq("token", token).execute()
            st.query_params.clear()
            st.session_state.usuario = None
            st.session_state.page = "inicio"
            st.rerun()

    if st.session_state.close_sidebar:
        st.session_state.close_sidebar = False
        st.iframe(f"""
            <script>
                setTimeout(function() {{
                    var doc = window.parent.document;
                    var sidebar = doc.querySelector('[data-testid="stSidebar"]');
                    if (sidebar) {{
                        var btn = sidebar.querySelector('button');
                        if (btn) btn.click();
                    }}
                }}, 300);
            </script>
            <!-- {time.time()} -->
        """, height=1)

    # Router
    page = st.session_state.page
    if   page == "inicio":       pagina_inicio()
    elif page == "reservar":     pagina_reservar()
    elif page == "mis_reservas": pagina_mis_reservas()
    elif page == "swing":        pagina_swing()


if __name__ == "__main__":
    main()
