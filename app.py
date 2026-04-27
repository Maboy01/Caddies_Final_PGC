#!/usr/bin/env python3
"""App de reserva de caddies y análisis de swing — Club Serrezuela."""
from __future__ import annotations

import os
import random
import sys
import time
import tempfile
from datetime import datetime, timedelta, date, time as dt_time
from pathlib import Path

import bcrypt
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


@st.cache_data(ttl=30)
def get_caddies() -> list:
    return get_supabase().table("caddies").select("*").eq("activo", True).execute().data


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False

def parse_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=None) if dt.tzinfo else dt

# ── Constantes ─────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_results" / "cnn_lstm" / "cnn_lstm_model.pt"

PRECIOS = {"1ra": 90_000, "2da": 70_000, "3ra": 60_000}

BADGE = {"1ra": "🥇 1ra Categoría", "2da": "🥈 2da Categoría", "3ra": "🥉 3ra Categoría"}
ESTADO_ICONO = {"activa": "🟢", "en_curso": "🟡", "cancelada": "🔴", "completada": "🔵"}

TURNOS = [
    f"{h:02d}:{m:02d}"
    for h in range(6, 11)
    for m in range(0, 60, 10)
    if not (h == 10 and m > 0)
]

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
    if "page"             not in st.session_state:
        st.session_state.page             = st.query_params.get("page", "inicio")
    if "caddie_pendiente" not in st.session_state:
        st.session_state.caddie_pendiente = None
    if "close_sidebar" not in st.session_state:
        st.session_state.close_sidebar = False
    if "confirmar_eliminar_caddie" not in st.session_state:
        st.session_state.confirmar_eliminar_caddie = None


# ── Helpers ─────────────────────────────────────────────────────────────────────

def db_error(e: Exception) -> None:
    st.error("Error de conexión con la base de datos. Intenta de nuevo.")
    st.caption(str(e))


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
            try:
                key = usuario_input.strip().lower()
                sb = get_supabase()
                session_data: dict = {"username": key}

                result = sb.table("usuarios").select("*").eq("username", key).execute()
                if result.data and verify_password(password_input, result.data[0]["password"]):
                    st.session_state.usuario = result.data[0]
                    session_data["rol"] = result.data[0].get("rol", "Socio")
                else:
                    caddie = sb.table("caddies").select("*").eq("username", key).eq("activo", True).execute()
                    if caddie.data and verify_password(password_input, caddie.data[0]["password"]):
                        data = caddie.data[0]
                        st.session_state.usuario = {
                            "username":  data["username"],
                            "nombre":    data["nombre"],
                            "rol":       "Caddie",
                            "caddie_id": data["id"],
                        }
                        session_data["rol"]       = "Caddie"
                        session_data["caddie_id"] = data["id"]

                if st.session_state.usuario:
                    expires = (datetime.now() + timedelta(hours=24)).isoformat()
                    session_data["expires_at"] = expires
                    session = sb.table("sessions").insert(session_data).execute()
                    try:
                        sb.table("sessions").delete().lt("expires_at", datetime.now().isoformat()).execute()
                    except Exception:
                        pass
                    st.query_params["token"] = session.data[0]["token"]
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos.")
            except Exception as e:
                db_error(e)


def pagina_inicio() -> None:
    st.markdown("## Bienvenido al Club Serrezuela")
    st.markdown("---")

    disponibles  = sum(1 for c in get_caddies() if c["disponible"])
    res_result   = get_supabase().table("reservas").select("id").eq("usuario_username", st.session_state.usuario["username"]).in_("estado", ["activa", "en_curso"]).execute()
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
            st.markdown(f"**{cop(PRECIOS['3ra'])}** por ronda")
    with col2:
        with st.container(border=True):
            st.markdown("#### 🥈 2da Categoría")
            st.write("1-2 años de experiencia")
            st.markdown(f"**{cop(PRECIOS['2da'])}** por ronda")
    with col3:
        with st.container(border=True):
            st.markdown("#### 🥇 1ra Categoría")
            st.write("3 o más años de experiencia")
            st.markdown(f"**{cop(PRECIOS['1ra'])}** por ronda")


def _tarjeta_caddie(caddie: dict) -> None:
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
            if caddie["disponible"]:
                if st.button("Reservar", key=f"res_{caddie['id']}", type="primary"):
                    st.session_state.caddie_pendiente = caddie
                    st.rerun()
            else:
                st.button("No disponible", key=f"nd_{caddie['id']}", disabled=True)


def _confirmar_reserva(caddie: dict) -> None:
    precio = PRECIOS[caddie["categoria"]]

    hora_actual = datetime.now().time()
    if hora_actual >= dt_time(18, 0) or hora_actual < dt_time(6, 0):
        st.warning("⚠️ Las reservas solo están disponibles entre las **6:00 AM** y las **6:00 PM**. Intenta de nuevo mañana.")
        if st.button("Volver", use_container_width=True):
            st.session_state.caddie_pendiente = None
            st.rerun()
        return

    hoy = date.today()
    min_fecha = hoy if hoy.weekday() != 0 else hoy + timedelta(days=1)
    fecha_juego = st.date_input(
        "Selecciona la fecha de inicio",
        value=min_fecha,
        min_value=hoy,
        max_value=hoy + timedelta(days=30),
    )

    dias = int(st.number_input("¿Por cuántos días?", min_value=1, max_value=3, value=1, step=1))
    fecha_fin = fecha_juego + timedelta(days=dias - 1)
    if dias > 1:
        st.info(f"El caddie estará reservado del **{fecha_juego.strftime('%d/%m/%Y')}** al **{fecha_fin.strftime('%d/%m/%Y')}** ({dias} días).")

    precio_total = precio * dias
    anticipo     = precio_total // 2

    st.markdown("**¿A qué hora es tu tee time?**")
    hora_juego = st.selectbox("Selecciona el tee time de salida", TURNOS)

    st.markdown("### Confirmar reserva")
    with st.container(border=True):
        st.markdown(f"**Caddie:** {caddie['nombre']}")
        st.markdown(f"**Categoría:** {BADGE[caddie['categoria']]}")
        st.markdown(f"**Calificación:** {estrellas(caddie['calificacion'])} {caddie['calificacion']}")
        st.markdown(f"**Días reservados:** {dias} × {cop(precio)} = **{cop(precio_total)}**")
        st.markdown(f"**Anticipo a pagar ahora (50%):** {cop(anticipo)}")
        st.info("Tienes **8 horas** para cancelar y recuperar el anticipo completo.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Confirmar y pagar", type="primary", use_container_width=True):
            dias_rango = [fecha_juego + timedelta(days=d) for d in range(dias)]
            if any(d.weekday() == 0 for d in dias_rango):
                st.error("El rango de días incluye un lunes. El campo está cerrado los lunes. Elige otra fecha.")
                return

            try:
                sb = get_supabase()
                username = st.session_state.usuario["username"]

                existente = sb.table("reservas").select("id").eq("usuario_username", username).eq("fecha_juego", fecha_juego.isoformat()).in_("estado", ["activa", "en_curso"]).execute()
                if existente.data:
                    st.error(f"Ya tienes una reserva activa para el {fecha_juego.strftime('%d/%m/%Y')}. Elige otro día.")
                    return

                slot_count = sb.table("reservas").select("id").eq("fecha_juego", fecha_juego.isoformat()).eq("hora_juego", hora_juego).in_("estado", ["activa", "en_curso"]).execute()
                if len(slot_count.data) >= 4:
                    st.error(f"El turno de las {hora_juego} del {fecha_juego.strftime('%d/%m/%Y')} ya está completo (4/4). Elige otro horario.")
                    return

                fecha      = datetime.now()
                limite     = fecha + timedelta(hours=8)
                resultado  = sb.table("reservas").insert({
                    "usuario_username":   username,
                    "caddie_id":          caddie["id"],
                    "precio_total":       precio_total,
                    "anticipo":           anticipo,
                    "fecha_reserva":      fecha.isoformat(),
                    "limite_cancelacion": limite.isoformat(),
                    "fecha_juego":        fecha_juego.isoformat(),
                    "hora_juego":         hora_juego,
                    "estado":             "activa",
                    "dias":               dias,
                }).execute()
                reserva_id = resultado.data[0]["id"]
                sb.table("caddies").update({"disponible": False}).eq("id", caddie["id"]).execute()
                get_caddies.clear()
                st.session_state.caddie_pendiente = None
                fecha_fin_str = f" al {fecha_fin.strftime('%d/%m/%Y')}" if dias > 1 else ""
                st.success(
                    f"Reserva #{reserva_id} confirmada del {fecha_juego.strftime('%d/%m/%Y')}{fecha_fin_str} a las {hora_juego}. "
                    f"Se cobró el anticipo de {cop(anticipo)}. "
                    f"{caddie['nombre']} ha sido notificado."
                )
                st.balloons()
                st.rerun()
            except Exception as e:
                db_error(e)
    with col2:
        if st.button("Volver", use_container_width=True):
            st.session_state.caddie_pendiente = None
            st.rerun()


def _solicitar_aleatorio(cat: str) -> None:
    disponibles = [c for c in get_caddies() if c["categoria"] == cat and c["disponible"]]
    if not disponibles:
        st.error(f"No hay caddies de {cat} categoría disponibles ahora.")
        return
    st.session_state.caddie_pendiente = random.choice(disponibles)
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
            c for c in get_caddies()
            if filtro == "Todas" or c["categoria"] == filtro
        ]
        for caddie in lista:
            _tarjeta_caddie(caddie)

    else:
        st.markdown("### Selecciona la categoría que deseas")
        col1, col2, col3 = st.columns(3)

        with col1:
            with st.container(border=True):
                st.markdown("#### 🥉 3ra Categoría")
                st.write("Sin experiencia mínima")
                st.markdown(f"**{cop(PRECIOS['3ra'])}** / ronda")
                if st.button("Solicitar 3ra", use_container_width=True):
                    _solicitar_aleatorio("3ra")
        with col2:
            with st.container(border=True):
                st.markdown("#### 🥈 2da Categoría")
                st.write("1-2 años de experiencia")
                st.markdown(f"**{cop(PRECIOS['2da'])}** / ronda")
                if st.button("Solicitar 2da", use_container_width=True):
                    _solicitar_aleatorio("2da")
        with col3:
            with st.container(border=True):
                st.markdown("#### 🥇 1ra Categoría")
                st.write("3+ años de experiencia")
                st.markdown(f"**{cop(PRECIOS['1ra'])}** / ronda")
                if st.button("Solicitar 1ra", use_container_width=True):
                    _solicitar_aleatorio("1ra")


@st.dialog("🏁 Finalizar turno")
def dialogo_calificacion(reserva_id: int, caddie: dict, saldo: int) -> None:
    st.markdown(f"¿Cómo estuvo **{caddie['nombre']}**?")
    calificacion = st.select_slider(
        "Tu calificación",
        options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        value=4.0,
        format_func=lambda x: f"{'★' * int(x)}{'½' if x % 1 else ''} ({x})",
    )
    st.markdown("---")
    st.markdown(f"**Saldo a pagar al caddie:** {cop(saldo)}")
    completar_pago = st.checkbox(f"Confirmar pago de {cop(saldo)} al caddie")
    if not completar_pago:
        st.caption("Debes confirmar el pago para finalizar el turno.")
    if st.button("Confirmar", type="primary", use_container_width=True, disabled=not completar_pago):
        try:
            sb = get_supabase()
            sb.table("reservas").update({
                "estado":             "completada",
                "calificacion_socio": calificacion,
                "pago_completado":    completar_pago,
            }).eq("id", reserva_id).execute()
            sb.table("caddies").update({
                "disponible": True,
                "rondas":     caddie["rondas"] + 1,
            }).eq("id", caddie["id"]).execute()
            get_caddies.clear()
            st.rerun()
        except Exception as e:
            db_error(e)


def pagina_mis_reservas() -> None:
    st.markdown("## 📋 Mis Reservas")

    sb       = get_supabase()
    username = st.session_state.usuario["username"]
    result   = sb.table("reservas").select("*, caddies(*)").eq("usuario_username", username).neq("estado", "cancelada").order("fecha_reserva", desc=True).execute()
    todas    = result.data

    if not todas:
        st.info("Aún no tienes reservas.")
        if st.button("Reservar un caddie", type="primary"):
            ir_a("reservar")
        return

    filtro = st.selectbox("Ver", ["Reservadas", "Completadas"], index=0, label_visibility="collapsed")

    activas, completadas = [], []
    for r in todas:
        if r["estado"] in ("activa", "en_curso"):
            activas.append(r)
        elif r["estado"] == "completada":
            completadas.append(r)

    reservas = activas if filtro == "Reservadas" else completadas

    if not reservas:
        st.info("No hay reservas en esta categoría.")
        return

    for i, reserva in enumerate(reservas):
        caddie = reserva.get("caddies")
        if not caddie:
            continue
        ahora      = datetime.now()
        estado     = reserva["estado"]
        activa    = estado == "activa"
        en_curso  = estado == "en_curso"
        en_tiempo = ahora < parse_dt(reserva["limite_cancelacion"])

        fecha_juego_str = reserva.get("fecha_juego")
        hora_juego_str  = (reserva.get("hora_juego") or "")[:5]
        es_hoy_o_antes  = False
        if fecha_juego_str:
            fj = date.fromisoformat(fecha_juego_str)
            es_hoy_o_antes = fj <= ahora.date()

        with st.container(border=True):
            icono = ESTADO_ICONO.get(estado, "⚪")
            st.markdown(f"**Reserva #{reserva['id']}** — {icono} {estado.replace('_', ' ').capitalize()}")
            st.markdown(f"**Caddie:** {caddie['nombre']} — {BADGE[caddie['categoria']]}")
            st.markdown(
                f"**Anticipo pagado:** {cop(reserva['anticipo'])}  |  "
                f"**Saldo pendiente:** {cop(reserva['precio_total'] - reserva['anticipo'])}"
            )
            hora_str = f" a las {hora_juego_str}" if hora_juego_str else ""
            dias_reserva = reserva.get("dias") or 1
            if dias_reserva > 1:
                fecha_fin_r = date.fromisoformat(fecha_juego_str) + timedelta(days=dias_reserva - 1) if fecha_juego_str else None
                fecha_fin_str = f" al {fecha_fin_r.strftime('%d/%m/%Y')}" if fecha_fin_r else ""
                st.markdown(f"**Fechas:** {fecha_juego_str or '—'}{fecha_fin_str}{hora_str} ({dias_reserva} días)")
            else:
                st.markdown(f"**Fecha de juego:** {fecha_juego_str or '—'}{hora_str}")
            st.markdown(f"**Reservado el:** {parse_dt(reserva['fecha_reserva']).strftime('%d/%m/%Y %H:%M')}")

            if activa and es_hoy_o_antes:
                st.info(f"Es hora del turno de las {hora_juego_str}.")
                if st.button("🟡 Iniciar turno", key=f"iniciar_{i}", use_container_width=True, type="primary"):
                    sb.table("reservas").update({"estado": "en_curso"}).eq("id", reserva["id"]).execute()
                    st.rerun()
            elif activa:
                if en_tiempo:
                    restante = parse_dt(reserva["limite_cancelacion"]) - ahora
                    h = int(restante.total_seconds() // 3600)
                    m = int((restante.total_seconds() % 3600) // 60)
                    st.warning(f"Puedes cancelar con reembolso durante {h}h {m}m más.")
                    if st.button(f"Cancelar reserva #{reserva['id']}", key=f"cancel_{i}"):
                        sb.table("reservas").update({"estado": "cancelada"}).eq("id", reserva["id"]).execute()
                        sb.table("caddies").update({"disponible": True}).eq("id", caddie["id"]).execute()
                        get_caddies.clear()
                        st.success(f"Reserva cancelada. Se reembolsarán {cop(reserva['anticipo'])}.")
                        st.rerun()
                else:
                    st.error("Ya no puedes cancelar con reembolso. El anticipo será transferido al caddie si no te presentas.")
                    if st.button(f"Cancelar sin reembolso #{reserva['id']}", key=f"cancel_nr_{i}"):
                        sb.table("reservas").update({"estado": "cancelada"}).eq("id", reserva["id"]).execute()
                        sb.table("caddies").update({"disponible": True}).eq("id", caddie["id"]).execute()
                        get_caddies.clear()
                        st.info(
                            f"Reserva cancelada. El anticipo de {cop(reserva['anticipo'])} "
                            f"fue transferido a {caddie['nombre']}."
                        )
                        st.rerun()
            elif en_curso:
                st.warning("El caddie está en turno activo.")
                extra = int(PRECIOS[caddie["categoria"]] * 0.30)
                st.caption(f"Total actual: {cop(reserva['precio_total'])}")
                col_a, col_b = st.columns(2)
                with col_a:
                    ultimo = reserva.get("ultimo_hoyo_extra")
                    if ultimo:
                        transcurrido = ahora - parse_dt(ultimo)
                        mins_restantes = 30 - int(transcurrido.total_seconds() // 60)
                        bloqueado = transcurrido.total_seconds() < 1800
                    else:
                        bloqueado = False
                        mins_restantes = 0

                    if bloqueado:
                        st.button(f"⛳ 9 hoyos más (+{cop(extra)})", key=f"hoyos_{i}", use_container_width=True, disabled=True)
                        st.caption(f"Disponible en {mins_restantes} min.")
                    else:
                        if st.button(f"⛳ 9 hoyos más (+{cop(extra)})", key=f"hoyos_{i}", use_container_width=True):
                            sb.table("reservas").update({
                                "precio_total":      reserva["precio_total"] + extra,
                                "ultimo_hoyo_extra": ahora.isoformat(),
                            }).eq("id", reserva["id"]).execute()
                            st.rerun()
                with col_b:
                    if st.button("🏁 Terminar turno", key=f"terminar_{i}", use_container_width=True, type="primary"):
                        saldo = reserva["precio_total"] - reserva["anticipo"]
                        dialogo_calificacion(reserva["id"], caddie, saldo)
            elif estado == "completada":
                cal  = reserva.get("calificacion_socio")
                pago = reserva.get("pago_completado")
                if cal:
                    st.success(f"Tu calificación: {estrellas(cal)} ({cal})")
                st.info("✅ Pago completado" if pago else "⏳ Pago pendiente")


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
        probs_ordenadas = sorted(todas.items(), key=lambda x: -x[1])

        if clase == "no_golf":
            st.warning(
                f"**No se detectó un swing de golf en el video** "
                f"(confianza: {confianza * 100:.1f}%). "
                "Asegúrate de subir un video con un swing completo y ángulo lateral claro."
            )
            st.markdown("**Distribución de probabilidades:**")
            for cls, prob in probs_ordenadas:
                etiqueta = CLUB_INFO.get(cls, (cls,))[0]
                st.progress(prob, text=f"{etiqueta}: {prob * 100:.1f}%")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Palo recomendado", nombre_club)
            st.metric("Confianza", f"{confianza * 100:.1f}%")
        with col2:
            st.markdown("**Distribución de probabilidades:**")
            for cls, prob in probs_ordenadas:
                etiqueta = CLUB_INFO.get(cls, (cls,))[0]
                st.progress(prob, text=f"{etiqueta}: {prob * 100:.1f}%")

        st.markdown("---")
        st.success(f"**{nombre_club}** — {descripcion}")


def pagina_caddie() -> None:
    usuario  = st.session_state.usuario
    caddie_id = usuario["caddie_id"]
    sb       = get_supabase()

    rows = sb.table("caddies").select("*").eq("id", caddie_id).execute().data
    if not rows:
        st.error("No se encontró la información del caddie. Contacta al administrador.")
        return
    caddie_data = rows[0]

    st.markdown(f"## 🏌️ Bienvenido, {caddie_data['nombre']}")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Categoría",    BADGE[caddie_data["categoria"]])
    col2.metric("Calificación", f"{caddie_data['calificacion']} ★")
    col3.metric("Rondas",       caddie_data["rondas"])

    st.markdown("---")
    st.markdown("### 📋 Tus reservas activas")

    result = sb.table("reservas").select("*").eq("caddie_id", caddie_id).in_("estado", ["activa", "en_curso"]).order("fecha_reserva", desc=False).execute()
    reservas = result.data

    if not reservas:
        st.info("No tienes reservas activas en este momento.")
        return

    usernames = list({r["usuario_username"] for r in reservas})
    nombres_map = {}
    if usernames:
        socios = sb.table("usuarios").select("username, nombre").in_("username", usernames).execute()
        nombres_map = {s["username"]: s["nombre"] for s in socios.data}

    ahora = datetime.now()
    for reserva in reservas:
        socio_nombre = nombres_map.get(reserva["usuario_username"], reserva["usuario_username"])
        fecha        = parse_dt(reserva["fecha_reserva"])
        limite       = parse_dt(reserva["limite_cancelacion"])
        en_tiempo    = ahora < limite

        with st.container(border=True):
            st.markdown(f"**Reserva #{reserva['id']}**")
            st.markdown(f"**Socio:** {socio_nombre}")
            hora = reserva.get("hora_juego") or ""
            hora_str = f" a las {hora[:5]}" if hora else ""
            st.markdown(f"**Fecha de juego:** {reserva.get('fecha_juego') or '—'}{hora_str}")
            st.markdown(f"**Reservado el:** {fecha.strftime('%d/%m/%Y %H:%M')}")
            st.markdown(f"**Pago total:** {cop(reserva['precio_total'])}  |  **Anticipo recibido:** {cop(reserva['anticipo'])}")
            if en_tiempo:
                restante = limite - ahora
                h = int(restante.total_seconds() // 3600)
                m = int((restante.total_seconds() % 3600) // 60)
                st.warning(f"El socio puede cancelar con reembolso durante {h}h {m}m más.")
            else:
                st.success("Ya no puede cancelar. El anticipo es tuyo si no se presenta.")


@st.dialog("➕ Nuevo caddie")
def dialogo_nuevo_caddie() -> None:
    with st.form("form_nuevo_caddie"):
        nombre      = st.text_input("Nombre completo")
        categoria   = st.selectbox("Categoría", ["1ra", "2da", "3ra"])
        experiencia = st.text_input("Experiencia")
        submitted   = st.form_submit_button("Crear caddie", type="primary", use_container_width=True)

    if submitted:
        if not all([nombre.strip(), experiencia.strip()]):
            st.error("Todos los campos son obligatorios.")
            return

        partes   = nombre.strip().lower().split()
        username = "".join(partes)
        password = partes[0] + "123"

        try:
            sb = get_supabase()
            existe = sb.table("caddies").select("id").eq("username", username).execute()
            if existe.data:
                st.error(f"Ya existe un caddie con el usuario '{username}'. Revisa el nombre.")
                return
            sb.table("caddies").insert({
                "nombre":       nombre.strip(),
                "categoria":    categoria,
                "experiencia":  experiencia.strip(),
                "username":     username,
                "password":     hash_password(password),
                "calificacion": 4.0,
                "rondas":       0,
                "disponible":   True,
                "activo":       True,
            }).execute()
            get_caddies.clear()
            st.success(f"Caddie **{nombre.strip()}** creado correctamente.")
            st.info(f"**Usuario:** `{username}`\n\n**Contraseña:** `{password}`")
        except Exception as e:
            db_error(e)


def _admin_metricas(reservas_all: list, caddies_all: list) -> None:
    hoy = date.today().isoformat()
    n_activas = n_completadas_hoy = 0
    for r in reservas_all:
        estado = r["estado"]
        if estado in ("activa", "en_curso"):
            n_activas += 1
        elif estado == "completada" and r.get("fecha_juego") == hoy:
            n_completadas_hoy += 1
    n_disponibles = sum(1 for c in caddies_all if c["disponible"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Reservas activas",    n_activas)
    c2.metric("Completadas hoy",     n_completadas_hoy)
    c3.metric("Caddies disponibles", n_disponibles)


def _admin_reservas(sb, reservas_all: list, caddies_map: dict, usuarios_map: dict) -> None:
    st.markdown("### 📋 Todas las reservas")
    filtro_estado      = st.selectbox("Filtrar por estado", ["Todas", "activa", "en_curso", "completada", "cancelada"])
    filtro_fecha_admin = st.date_input("Filtrar por fecha", value=None, format="DD/MM/YYYY", key="admin_fecha")

    reservas_filtradas = reservas_all if filtro_estado == "Todas" else [r for r in reservas_all if r["estado"] == filtro_estado]
    if filtro_fecha_admin:
        reservas_filtradas = [r for r in reservas_filtradas if r.get("fecha_juego") == filtro_fecha_admin.isoformat()]
    reservas_filtradas = sorted(reservas_filtradas, key=lambda r: r["fecha_reserva"], reverse=True)

    for r in reservas_filtradas:
        icono = ESTADO_ICONO.get(r["estado"], "⚪")
        with st.container(border=True):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**Reserva #{r['id']}** — {icono} {r['estado'].replace('_',' ').capitalize()}")
                st.markdown(f"**Socio:** {usuarios_map.get(r['usuario_username'], r['usuario_username'])}  |  **Caddie:** {caddies_map.get(r['caddie_id'], r['caddie_id'])}")
                hora = (r.get("hora_juego") or "")[:5]
                st.markdown(f"**Fecha:** {r.get('fecha_juego') or '—'}{f' a las {hora}' if hora else ''}  |  **Total:** {cop(r['precio_total'])}  |  **Anticipo:** {cop(r['anticipo'])}")
            with col_b:
                if r["estado"] in ("activa", "en_curso"):
                    if st.button("Cancelar", key=f"adm_cancel_{r['id']}", type="secondary"):
                        sb.table("reservas").update({"estado": "cancelada"}).eq("id", r["id"]).execute()
                        sb.table("caddies").update({"disponible": True}).eq("id", r["caddie_id"]).execute()
                        get_caddies.clear()
                        st.rerun()


def _admin_caddies(sb, caddies_all: list) -> None:
    col_titulo, col_btn = st.columns([4, 1])
    col_titulo.markdown("### 🏌️ Gestión de caddies")
    with col_btn:
        st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
        if st.button("➕ Nuevo caddie", use_container_width=True, type="primary"):
            dialogo_nuevo_caddie()

    for c in sorted(caddies_all, key=lambda x: x["nombre"]):
        with st.container(border=True):
            col_info, col_toggle, col_del = st.columns([4, 1, 1])
            with col_info:
                st.markdown(f"**{c['nombre']}** — {BADGE[c['categoria']]}")
                st.markdown(f"{estrellas(c['calificacion'])} `{c['calificacion']}`  |  {c['rondas']} rondas")
            with col_toggle:
                estado_actual = "✅ Disponible" if c["disponible"] else "❌ No disponible"
                label_btn     = "Deshabilitar" if c["disponible"] else "Habilitar"
                if st.button(label_btn, key=f"adm_toggle_{c['id']}"):
                    sb.table("caddies").update({"disponible": not c["disponible"]}).eq("id", c["id"]).execute()
                    get_caddies.clear()
                    st.rerun()
                st.caption(estado_actual)
            with col_del:
                if st.session_state.confirmar_eliminar_caddie == c["id"]:
                    st.caption("¿Confirmar?")
                    col_si, col_no = st.columns(2)
                    with col_si:
                        if st.button("Sí", key=f"adm_del_si_{c['id']}", type="primary"):
                            activa = sb.table("reservas").select("id").eq("caddie_id", c["id"]).in_("estado", ["activa", "en_curso"]).execute()
                            if activa.data:
                                st.error("Tiene reservas activas. Cancélalas primero.")
                            else:
                                sb.table("caddies").update({"activo": False, "disponible": False}).eq("id", c["id"]).execute()
                                get_caddies.clear()
                            st.session_state.confirmar_eliminar_caddie = None
                            st.rerun()
                    with col_no:
                        if st.button("No", key=f"adm_del_no_{c['id']}"):
                            st.session_state.confirmar_eliminar_caddie = None
                            st.rerun()
                else:
                    if st.button("Eliminar", key=f"adm_del_{c['id']}", type="secondary"):
                        st.session_state.confirmar_eliminar_caddie = c["id"]
                        st.rerun()


def pagina_admin() -> None:
    st.markdown("## 🛠️ Panel de Administración")
    sb = get_supabase()

    reservas_all  = sb.table("reservas").select("*").execute().data
    caddies_all   = get_caddies()
    todos_caddies = sb.table("caddies").select("id, nombre").execute().data
    caddies_map   = {c["id"]: c["nombre"] for c in todos_caddies}
    usuarios_map  = {u["username"]: u["nombre"] for u in sb.table("usuarios").select("username, nombre").execute().data}

    _admin_metricas(reservas_all, caddies_all)
    st.markdown("---")
    _admin_reservas(sb, reservas_all, caddies_map, usuarios_map)
    st.markdown("---")
    _admin_caddies(sb, caddies_all)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Club Serrezuela — Caddies",
        page_icon="⛳",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        "<style>[data-testid='stToolbar'] {display: none;}</style>",
        unsafe_allow_html=True,
    )

    init_state()

    if st.session_state.usuario is None:
        token = st.query_params.get("token")
        if token:
            sb = get_supabase()
            session = sb.table("sessions").select("username, rol, caddie_id").eq("token", token).gt("expires_at", datetime.now().isoformat()).execute()
            if session.data:
                s = session.data[0]
                if s.get("rol") == "Caddie" and s.get("caddie_id"):
                    caddie = sb.table("caddies").select("*").eq("id", s["caddie_id"]).eq("activo", True).execute()
                    if caddie.data:
                        data = caddie.data[0]
                        st.session_state.usuario = {
                            "username":  data["username"],
                            "nombre":    data["nombre"],
                            "rol":       "Caddie",
                            "caddie_id": data["id"],
                        }
                        st.rerun()
                else:
                    result = sb.table("usuarios").select("*").eq("username", s["username"]).execute()
                    if result.data:
                        st.session_state.usuario = result.data[0]
                        st.rerun()

    if st.session_state.usuario is None:
        pagina_login()
        return

    usuario   = st.session_state.usuario
    es_caddie = usuario.get("rol") == "Caddie"
    es_admin  = usuario.get("rol") == "Admin"

    # Sidebar
    with st.sidebar:
        st.markdown("## ⛳ Club Serrezuela")
        st.markdown("---")

        if es_admin:
            paginas = [("🛠️ Panel Admin", "admin")]
        elif not es_caddie:
            paginas = [
                ("🏠 Inicio",          "inicio"),
                ("🏌️ Reservar Caddie",  "reservar"),
                ("📋 Mis Reservas",     "mis_reservas"),
                ("🎯 Analiza tu Swing", "swing"),
            ]
        else:
            paginas = []

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
    if es_admin:
        pagina_admin()
    elif es_caddie:
        pagina_caddie()
    else:
        page = st.session_state.page
        if   page == "inicio":       pagina_inicio()
        elif page == "reservar":     pagina_reservar()
        elif page == "mis_reservas": pagina_mis_reservas()
        elif page == "swing":        pagina_swing()
        else:                        ir_a("inicio")


if __name__ == "__main__":
    main()
