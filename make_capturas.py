"""
make_capturas.py - Genera las capturas del reporte con el aspecto del tema oscuro de Visual Studio Code
(editor en tarjeta redondeada, barra de estado oscura, llaves coloreadas), a partir del codigo y de
los logs REALES de cada ejecucion (carpeta logs/).
Autor: Kevin del Jesus Gonzalez Maas - Sistemas Inteligentes - FI-UAC
"""
import os
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token

RAIZ = os.path.dirname(os.path.abspath(__file__))
D = "/usr/share/fonts/truetype/dejavu/"
CX = "/usr/share/fonts/truetype/crosextra/"
ESC, PX = 2, 17
E = lambda v: int(v * ESC)
mono = ImageFont.truetype(D + "DejaVuSansMono.ttf", E(PX))
CW = mono.getlength("M")
LH = int(E(PX) * 1.5)
def ui(n, b=False):
    ruta = CX + ("Carlito-Bold.ttf" if b else "Carlito-Regular.ttf")
    if not os.path.exists(ruta): ruta = D + ("DejaVuSans-Bold.ttf" if b else "DejaVuSans.ttf")
    return ImageFont.truetype(ruta, E(n + 1.5))

def uis(n, b=False):                                   # fuente con simbolos (⌄ ✕ ⊗ ⚠ ♪)
    return ImageFont.truetype(D + ("DejaVuSans-Bold.ttf" if b else "DejaVuSans.ttf"), E(n))

K = dict(chrome="#191A1B", card="#121314", borde="#2B2D2E", tab="#2C2D2E", linea="#242526", texto="#BBBEBF",
         gris="#8C8C8C", icono="#8C8C8C", activo="#CCCCCC", guia="#2E3032", term="#CCCCCC")
ROSA, AZUL, CAD, COM = "#F28CE6", "#79C0FF", "#A5D6FF", "#8B949E"
LLAVES = ["#FFD60A", "#DA70D6", "#179FFF"]

def lineas_coloreadas(texto):
    toks, lineas, actual, prof = list(lex(texto, PythonLexer())), [], [], 0
    for tok, val in toks:
        if tok in Token.Comment: col = COM
        elif tok in Token.Literal.String.Interpol: col = AZUL
        elif tok in Token.Literal.String: col = CAD
        elif tok in Token.Literal.Number: col = AZUL
        elif tok in Token.Keyword.Constant: col = AZUL
        elif tok in Token.Keyword or tok in Token.Operator.Word: col = ROSA
        elif tok in Token.Name and val.startswith("__") and val.endswith("__"): col = AZUL
        elif tok in Token.Punctuation and val in "([{":
            col = LLAVES[prof % 3]; prof += 1
        elif tok in Token.Punctuation and val in ")]}":
            prof = max(prof - 1, 0); col = LLAVES[prof % 3]
        else: col = K["texto"]
        partes = val.split("\n")
        for k, p in enumerate(partes):
            if k > 0: lineas.append(actual); actual = []
            if p: actual.append((p, col))
    lineas.append(actual)
    return lineas

# ---------------------------------------------------------------- iconos y elementos de la interfaz
def lupa(d, x, y, col, s=1.0):
    d.ellipse([x + E(8*s), y + E(5*s), x + E(20*s), y + E(17*s)], outline=col, width=E(1.5))
    d.line([x + E(9*s), y + E(16*s), x + E(4*s), y + E(22*s)], fill=col, width=E(2))

def icono_act(d, x, y, kind, activo):
    col = K["activo"] if activo else K["icono"]; s = E(1.5)
    if kind == "files":
        d.rounded_rectangle([x + E(5), y + E(6), x + E(17), y + E(24)], radius=E(2), outline=col, width=s)
        d.rounded_rectangle([x + E(11), y + E(11), x + E(23), y + E(29)], radius=E(2), outline=col, width=s, fill=K["chrome"])
    elif kind == "search": lupa(d, x + E(3), y + E(3), col, 1.1)
    elif kind == "git":
        for (cx, cy) in [(9, 7), (9, 26), (22, 14)]:
            d.ellipse([x + E(cx - 3), y + E(cy - 3), x + E(cx + 3), y + E(cy + 3)], outline=col, width=s)
        d.line([x + E(9), y + E(10), x + E(9), y + E(23)], fill=col, width=s)
        d.line([x + E(22), y + E(17), x + E(9), y + E(23)], fill=col, width=s)
    elif kind == "run":
        d.polygon([(x + E(7), y + E(6)), (x + E(7), y + E(22)), (x + E(19), y + E(14))], outline=col, width=s)
        d.ellipse([x + E(14), y + E(19), x + E(25), y + E(29)], outline=col, width=s)
    elif kind == "ext":
        for (dx, dy) in [(0, 0), (10, 0), (0, 10)]:
            d.rectangle([x + E(6 + dx), y + E(8 + dy), x + E(14 + dx), y + E(16 + dy)], outline=col, width=s)
        d.rectangle([x + E(17), y + E(4), x + E(25), y + E(12)], outline=col, width=s)

def icono_py(d, x, y):
    d.rounded_rectangle([x, y, x + E(8), y + E(8)], radius=E(3), fill="#3B78B8")
    d.rounded_rectangle([x + E(5), y + E(5), x + E(13), y + E(13)], radius=E(3), fill="#E7C547")

def barra_titulo(d, W):
    d.rectangle([0, 0, W, E(38)], fill=K["chrome"])
    d.polygon([(E(15), E(12)), (E(27), E(8)), (E(27), E(30)), (E(15), E(26))], fill="#3B8EEA")
    x = E(46)
    for m in ["File", "Edit", "Selection", "View", "Go", "Run", "···"]:
        d.text((x, E(19)), m, font=ui(13), fill=K["activo"], anchor="lm"); x += int(ui(13).getlength(m)) + E(18)
    cx, w = W // 2, E(250)
    d.text((cx - w // 2 - E(48), E(19)), "←    →", font=ui(14), fill="#5A5C5D", anchor="lm")
    d.rounded_rectangle([cx - w // 2, E(8), cx + w // 2, E(30)], radius=E(5), outline=K["borde"], width=E(1), fill=K["chrome"])
    lupa(d, cx - w // 2 + E(8), E(8), K["gris"], 0.55)
    d.text((cx - w // 2 + E(30), E(19)), "Search", font=ui(12), fill=K["gris"], anchor="lm")
    for i in range(4):
        d.rectangle([W - E(210) + i * E(30), E(12), W - E(210) + i * E(30) + E(14), E(26)], outline=K["icono"], width=E(1.3))
    for i, g in enumerate(["—", "☐", "✕"]):
        d.text((W - E(100) + i * E(34), E(19)), g, font=uis(11), fill=K["activo"], anchor="mm")

def barra_actividad(d, alto):
    for i, k in enumerate(["files", "search", "git", "run", "ext"]):
        icono_act(d, E(10), E(48) + i * E(46), k, i == 0)
    d.ellipse([E(14), alto - E(112), E(36), alto - E(90)], outline=K["icono"], width=E(1.5))
    d.ellipse([E(21), alto - E(108), E(29), alto - E(100)], outline=K["icono"], width=E(1.3))
    d.ellipse([E(13), alto - E(66), E(35), alto - E(44)], outline=K["icono"], width=E(1.5))
    d.ellipse([E(19), alto - E(60), E(29), alto - E(50)], outline=K["icono"], width=E(1.5))

ARBOL = [("d", "capturas"), ("d", "datos"), ("d", "figuras"), ("d", "logs"), ("f", "comun.py"),
         ("f", "ej1_clasificacion_churn.py"), ("f", "ej2_regresion_concreto.py"),
         ("f", "ej3_series_tiempo_pm25.py"), ("f", "make_capturas.py"), ("f", "README.md"),
         ("f", "requirements.txt"), ("f", "RNA_Practica_Colab.ipynb")]
def col_ext(n): return {"py": "#4B8BBE", "md": "#6CB6FF", "txt": "#B5B8BA", "ipynb": "#E37933"}.get(n.rsplit(".", 1)[-1], "#CCCCCC")

def explorador(d, x0, alto, activo):
    d.text((x0 + E(20), E(60)), "EXPLORER", font=ui(11), fill=K["gris"], anchor="lm")
    d.text((x0 + E(206), E(60)), "···", font=ui(12, True), fill=K["gris"], anchor="rm")
    d.text((x0 + E(10), E(84)), "⌄", font=uis(11, True), fill=K["activo"], anchor="lm")
    d.text((x0 + E(26), E(84)), "PRACTICA_RNA_PYTHON", font=ui(11, True), fill=K["activo"], anchor="lm")
    y = E(96)
    for tipo, nom in ARBOL:
        if nom == activo: d.rounded_rectangle([x0 + E(4), y, x0 + E(222), y + E(22)], radius=E(4), fill=K["tab"])
        if tipo == "d":
            d.text((x0 + E(18), y + E(11)), "›", font=ui(14, True), fill=K["activo"], anchor="lm")
            d.text((x0 + E(34), y + E(11)), nom, font=ui(12), fill=K["texto"], anchor="lm")
        else:
            d.rounded_rectangle([x0 + E(24), y + E(5), x0 + E(34), y + E(17)], radius=E(2), fill=col_ext(nom))
            d.text((x0 + E(42), y + E(11)), nom, font=ui(12), fill="#FFFFFF" if nom == activo else K["texto"], anchor="lm")
        y += E(22)

def tarjeta(d, x0, y0, x1, y1):
    d.rounded_rectangle([x0, y0, x1, y1], radius=E(8), fill=K["card"], outline=K["borde"], width=E(1))

def pestanas(d, x0, y0, x1, activo):
    n = activo
    w = int(ui(12).getlength(n)) + E(64)
    d.rounded_rectangle([x0 + E(6), y0 + E(6), x0 + E(6) + w, y0 + E(40)], radius=E(7), fill=K["tab"])
    icono_py(d, x0 + E(18), y0 + E(17))
    d.text((x0 + E(38), y0 + E(24)), n, font=ui(12), fill="#FFFFFF", anchor="lm")
    d.text((x0 + E(6) + w - E(18), y0 + E(23)), "✕", font=uis(9), fill=K["activo"], anchor="mm")
    d.text((x1 - E(14), y0 + E(24)), "···", font=ui(12, True), fill=K["activo"], anchor="rm")
    d.rectangle([x1 - E(54), y0 + E(18), x1 - E(40), y0 + E(30)], outline=K["activo"], width=E(1.2))
    ruta = "C:  ›  Users  ›  DesarrolloIngenieria  ›  Desktop  ›  Practica_RNA_Python  ›  "
    d.text((x0 + E(14), y0 + E(54)), ruta, font=ui(11.5), fill=K["gris"], anchor="lm")
    xb = x0 + E(14) + int(ui(11.5).getlength(ruta))
    icono_py(d, xb, y0 + E(48)); d.text((xb + E(20), y0 + E(54)), n, font=ui(11.5), fill=K["gris"], anchor="lm")

def estado(d, W, alto, ln):
    y = alto - E(13)
    d.text((E(14), y), "><    ⊗ 0   ⚠ 0", font=uis(10.5), fill=K["texto"], anchor="lm")
    d.text((W - E(14), y), f"Ln {ln}, Col 1     Spaces: 4     UTF-8     LF     {{}} Python     Python 3.12      ♪", font=uis(10.5), fill=K["texto"], anchor="rm")

def finalizar(img, nombre):
    w, h = img.size
    img = img.resize((w // ESC, h // ESC), Image.LANCZOS)
    ruta = os.path.join(RAIZ, "capturas", nombre); img.save(ruta, optimize=True)
    print("[captura]", os.path.relpath(ruta, RAIZ), img.size)

def texto_codigo(d, lineas, x, y, ini, x_fin, resaltar=None):
    for i, ln in enumerate(lineas):
        yy = y + i * LH
        if resaltar is not None and i == resaltar:
            d.rectangle([x - E(8), yy, x_fin, yy + LH], fill=K["linea"])
        d.text((x - E(22), yy), f"{ini + i}", font=mono, fill="#E6E6E6" if i == resaltar else "#6E7072", anchor="ra")
        if ln and ln[0][0].startswith(" "):
            lead = len(ln[0][0]) - len(ln[0][0].lstrip(" "))
            for g in range(0, lead, 4):
                d.line([x + int(g * CW), yy, x + int(g * CW), yy + LH], fill=K["guia"], width=E(1))
        cx = x
        for txt, col in ln:
            d.text((cx, yy), txt, font=mono, fill=col); cx += CW * len(txt)

def minimapa(d, x0, y0, ancho, alto, lineas):
    y = y0 + E(6)
    for ln in lineas:
        if y > alto - E(6): break
        x = x0 + E(6)
        for txt, col in ln:
            w = len(txt) * E(0.6)
            if txt.strip() and x + w < x0 + ancho - E(4): d.rectangle([x, y, x + w, y + E(2)], fill=col)
            x += w
        y += E(3.6)

def captura_codigo(archivo, ini, fin, nombre):
    sel = open(os.path.join(RAIZ, archivo), encoding="utf-8").read().split("\n")[ini - 1:fin]
    while sel and not sel[-1].strip(): sel.pop()
    sel.append("")                                         # el cursor queda en una linea vacia final, como en tu captura
    lineas = lineas_coloreadas("\n".join(sel))[:len(sel)]
    n, anchoc = len(sel), max(len(l) for l in sel)
    act, lat, mini, goteo = E(50), 0, E(64), E(64)
    cx0 = act + lat + E(3)
    W = cx0 + goteo + int(anchoc * CW) + E(40) + mini + E(10)
    y_ed = E(40) + E(70)
    alto = y_ed + n * LH + E(24) + E(32)
    img = Image.new("RGB", (W, alto), K["chrome"]); d = ImageDraw.Draw(img)
    barra_titulo(d, W); barra_actividad(d, alto)
    x1, y1 = W - E(8), alto - E(32)
    tarjeta(d, cx0, E(40), x1, y1)
    pestanas(d, cx0, E(40), x1, archivo)
    texto_codigo(d, lineas, cx0 + goteo, y_ed + E(6), ini, x1 - mini - E(4), resaltar=n - 1)
    minimapa(d, x1 - mini, y_ed, mini, y1, lineas)
    estado(d, W, alto, ini + n - 1)
    finalizar(img, nombre)

PROMPT = "PS C:\\Users\\DesarrolloIngenieria\\Desktop\\Practica_RNA_Python> "
def captura_consola(archivo, log, ini, fin, nombre, comando, lineas_top=9):
    L = open(os.path.join(RAIZ, "logs", log), encoding="utf-8").read().rstrip("\n").split("\n")[ini - 1:fin]
    cab = open(os.path.join(RAIZ, archivo), encoding="utf-8").read().split("\n")[:lineas_top]
    cod = lineas_coloreadas("\n".join(cab))[:lineas_top]
    todo = ([PROMPT + comando] if comando else []) + L
    anchoc = max(max(len(l) for l in todo), max(len(l) for l in cab))
    act, lat, goteo = E(50), 0, E(64)
    cx0 = act + lat + E(3)
    W = cx0 + goteo + int(anchoc * CW) + E(50)
    y_ed = E(40) + E(70)
    h_ed = lineas_top * LH + E(18)
    y_p0 = y_ed + h_ed + E(6)
    alto = y_p0 + E(46) + len(todo) * LH + E(20) + E(32)
    img = Image.new("RGB", (W, alto), K["chrome"]); d = ImageDraw.Draw(img)
    barra_titulo(d, W); barra_actividad(d, alto)
    x1, y1 = W - E(8), alto - E(32)
    tarjeta(d, cx0, E(40), x1, y_ed + h_ed)
    pestanas(d, cx0, E(40), x1, archivo)
    texto_codigo(d, cod, cx0 + goteo, y_ed + E(6), 1, x1 - E(10))
    tarjeta(d, cx0, y_p0, x1, y1)
    x = cx0 + E(20)
    for t in ["PROBLEMS", "OUTPUT", "DEBUG CONSOLE", "TERMINAL", "PORTS"]:
        on = t == "TERMINAL"
        d.text((x, y_p0 + E(20)), t, font=ui(11.5), fill="#FFFFFF" if on else K["gris"], anchor="lm")
        w = int(ui(11.5).getlength(t))
        if on: d.line([x, y_p0 + E(34), x + w, y_p0 + E(34)], fill=K["activo"], width=E(1.5))
        x += w + E(26)
    d.text((x1 - E(16), y_p0 + E(20)), "powershell     +  ⌄   ⌃   ✕", font=uis(10.5), fill=K["activo"], anchor="rm")
    y = y_p0 + E(46)
    for l in todo:
        d.text((cx0 + goteo - E(24), y), l, font=mono, fill=K["term"]); y += LH
    estado(d, W, alto, lineas_top)
    finalizar(img, nombre)

if __name__ == "__main__":
    os.makedirs(os.path.join(RAIZ, "capturas"), exist_ok=True)
    e1, e2, e3 = "ej1_clasificacion_churn.py", "ej2_regresion_concreto.py", "ej3_series_tiempo_pm25.py"
    captura_codigo(e1, 27, 46, "ej1_codigo_datos.png")
    captura_codigo(e1, 47, 80, "ej1_codigo_entrenamiento.png")
    captura_codigo(e1, 81, 116, "ej1_codigo_evaluacion.png")
    captura_consola(e1, "ej1.txt", 1, 14, "ej1_consola_datos.png", "python ej1_clasificacion_churn.py")
    captura_consola(e1, "ej1.txt", 15, 44, "ej1_consola_ajuste.png", "")
    captura_consola(e1, "ej1.txt", 45, 84, "ej1_consola_evaluacion.png", "")
    captura_codigo(e2, 30, 46, "ej2_codigo_datos.png")
    captura_codigo(e2, 47, 72, "ej2_codigo_validacion_cruzada.png")
    captura_codigo(e2, 73, 105, "ej2_codigo_entrenamiento.png")
    captura_consola(e2, "ej2.txt", 1, 26, "ej2_consola_datos.png", "python ej2_regresion_concreto.py")
    captura_consola(e2, "ej2.txt", 27, 69, "ej2_consola_evaluacion.png", "")
    captura_codigo(e3, 25, 62, "ej3_codigo_ventanas.png")
    captura_codigo(e3, 63, 112, "ej3_codigo_entrenamiento.png")
    captura_codigo(e3, 138, 172, "ej3_codigo_evaluacion.png")
    captura_consola(e3, "ej3.txt", 1, 20, "ej3_consola_datos.png", "python ej3_series_tiempo_pm25.py")
    captura_consola(e3, "ej3.txt", 20, 44, "ej3_consola_ajuste.png", "")
    captura_consola(e3, "ej3.txt", 45, 74, "ej3_consola_evaluacion.png", "")
