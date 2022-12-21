import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt
import PySimpleGUI as sg


# Función de inferencia
def inferir_alergia(respuesta1, respuesta2, respuesta3):
    # Escalas de parametrización
    x_estornudos = np.arange(0, 10.1, 0.1)
    x_ojos_llorosos = np.arange(0, 10.1, 0.1)
    x_picazon = np.arange(0, 10.1, 0.1)
    x_alergia = np.arange(0, 10.1, 0.1)

    # Funciones de pertenencia estornudos
    estornudos_bajo = fuzz.trapmf(x_estornudos, [0, 0, 2, 4])
    estornudos_medio = fuzz.trimf(x_estornudos, [2, 5, 8])
    estornudos_alto = fuzz.trapmf(x_estornudos, [6, 8, 10, 10])

    # Funciones de pertenencia ojos llorosos
    ojos_llorosos_bajo = fuzz.trapmf(x_ojos_llorosos, [0, 0, 2, 4])
    ojos_llorosos_medio = fuzz.trimf(x_ojos_llorosos, [2, 5, 8])
    ojos_llorosos_alto = fuzz.trapmf(x_ojos_llorosos, [6, 8, 10, 10])

    # Funciones de pertenencia picazón
    picazon_bajo = fuzz.trapmf(x_ojos_llorosos, [0, 0, 2, 4])
    picazon_medio = fuzz.trimf(x_picazon, [2, 5, 8])
    picazon_alto = fuzz.trapmf(x_ojos_llorosos, [6, 8, 10, 10])

    # Funciones de pertenencia alergia
    alergia_bajo = fuzz.trapmf(x_picazon, [0, 0, 2, 4])
    alergia_medio = fuzz.trimf(x_picazon, [2, 3.5, 8])
    alergia_alto = fuzz.trapmf(x_picazon, [6.5, 9, 10, 10])

    figura = plt.figure(figsize = (12.8, 7.2))
    fila = 3
    columna = 2

    plt.subplot(fila, columna, 1)
    plt.plot(x_estornudos, estornudos_bajo, 'b', linewidth=1.5, label='Bajo')
    plt.plot(x_estornudos, estornudos_medio, 'g', linewidth=1.5, label='Medio')
    plt.plot(x_estornudos, estornudos_alto, 'r', linewidth=1.5, label='Alto')
    plt.title('Estornudos')
    plt.legend()

    plt.subplot(fila, columna, 2)
    plt.plot(x_ojos_llorosos, ojos_llorosos_bajo, 'b', linewidth=1.5, label='Bajo')
    plt.plot(x_ojos_llorosos, ojos_llorosos_medio, 'g', linewidth=1.5, label='Medio')
    plt.plot(x_ojos_llorosos, ojos_llorosos_alto, 'r', linewidth=1.5, label='Alto')
    plt.title('Ojos llorosos')
    plt.legend()

    plt.subplot(fila, columna, 3)
    plt.plot(x_picazon, picazon_bajo, 'b', linewidth=1.5, label='Bajo')
    plt.plot(x_picazon, picazon_medio, 'g', linewidth=1.5, label='Medio')
    plt.plot(x_picazon, picazon_alto, 'r', linewidth=1.5, label='Alto')
    plt.title('Picazón')
    plt.legend()

    
    plt.subplot(fila, columna, 4)
    plt.plot(x_alergia, alergia_bajo, 'b', linewidth=1.5, label='Bajo')
    plt.plot(x_alergia, alergia_medio, 'g', linewidth=1.5, label='Medio')
    plt.plot(x_alergia, alergia_alto, 'r', linewidth=1.5, label='Alto')
    plt.title('Alergia')
    plt.legend()


    # Fuzzificación de los valores de entrada para los estornudos
    estornudos_bajo = fuzz.interp_membership(x_estornudos, estornudos_bajo, respuesta1)
    estornudos_medio = fuzz.interp_membership(x_estornudos, estornudos_medio, respuesta1)
    estornudos_alto = fuzz.interp_membership(x_estornudos, estornudos_alto, respuesta1)

    # Fuzzificación de los valores de entrada para los ojos llorosos
    ojos_llorosos_bajo = fuzz.interp_membership(x_ojos_llorosos, ojos_llorosos_bajo, respuesta2)
    ojos_llorosos_medio = fuzz.interp_membership(x_ojos_llorosos, ojos_llorosos_medio, respuesta2)
    ojos_llorosos_alto = fuzz.interp_membership(x_ojos_llorosos, ojos_llorosos_alto, respuesta2)

    # Fuzzificación de los valores de entrada para la picazón
    picazon_bajo = fuzz.interp_membership(x_picazon, picazon_bajo, respuesta3)
    picazon_medio = fuzz.interp_membership(x_picazon, picazon_medio, respuesta3)
    picazon_alto = fuzz.interp_membership(x_picazon, picazon_alto, respuesta3)


    # Reglas
    # Regla 1: Si los estornudos son bajos, los ojos llorosos son bajos y la picazon es baja, entonces la alergia es baja
    regla_1 = np.fmin(estornudos_bajo, np.fmin(ojos_llorosos_bajo, picazon_bajo))
    # Regla 2: Si lo estornudos son medios, los ojos llorosos son medios y la picazon es media, entonces la alergia es media
    regla_2 = np.fmin(estornudos_medio, np.fmin(ojos_llorosos_medio, picazon_medio))
    # Regla 3: Si los estornudos son altos, los ojos llorosos son altos y la picazon es alta, entonces la alergia es alta
    regla_3 = np.fmin(estornudos_alto, np.fmin(ojos_llorosos_alto, picazon_alto))
    # Regla 4: Si los ojos llorosos o la picazon son altos, entonces la alergia es alta
    regla_4 = np.fmax(ojos_llorosos_alto, picazon_alto)
    # Regla 5: Si los estornudos son medios, los ojos llorosos son medios o la picazon es media, entonces la alergia es media
    regla_5 = np.fmin(estornudos_medio, np.fmax(ojos_llorosos_medio, picazon_medio))
    # Regla 6: Si los estornudos son altos, los ojos llorosos son bajos y la picazon es baja, entonces la alergia es baja
    regla_6 = np.fmin(estornudos_alto, np.fmin(ojos_llorosos_bajo, picazon_bajo))
    # Regla 7: Si los estornudos son bajos, los ojos llorosos son altos y la picazon es alta, entonces la alergia es alta
    regla_7 = np.fmin(estornudos_bajo, np.fmin(ojos_llorosos_alto, picazon_alto))
    # Regla 8: Si la picazon es alta y los ojos llorosos son medios, entonces la alergia es alta
    regla_8 = np.fmin(picazon_alto, ojos_llorosos_medio)
    # Regla 9: Si la picazon es baja y los ojos llorosos son medios, entonces la alergia es baja
    regla_9 = np.fmin(picazon_bajo, ojos_llorosos_medio)
    # Regla 10: Si la los ojos llorosos es baja o la picazon es baja, entonces la alergia es baja
    regla_10 = np.fmax(picazon_bajo, ojos_llorosos_bajo)
    # Regla 11: Si los ojos llorosos son medios o la picazon es media, entonces la alergia es media
    regla_11 = np.fmax(picazon_medio, ojos_llorosos_medio)
    # Regla 12: Si los estornudos son medios o la picazon es media, entonces la alergia es media
    regla_12 = np.fmax(estornudos_medio, picazon_medio)
    # Regla 13: Si los estornudos son altos, la alergia es alta
    regla_13 = estornudos_alto
    # Regla 14: Si los ojos llorosos son altos, la alergia es alta
    regla_14 = ojos_llorosos_alto
    # Regla 15: Si los ojos llorosos son altos y la picazon es media, entonces la alergia es alta
    regla_15 = np.fmin(ojos_llorosos_alto, picazon_medio)
    # Regla 16: Si los estornudos son medios y los ojos llorosos son altos, entonces la alergia es alta
    regla_16 = np.fmin(estornudos_medio, ojos_llorosos_alto)
    # Regla 17: Si los ojos llorosos son bajos y la picazon es alta o los estornudos son altos, entonces la alergia es media
    regla_17 = np.fmin(ojos_llorosos_bajo, np.fmax(picazon_alto, estornudos_alto))

    # Activación de las reglas
    alergia_baja_1 = np.fmin(regla_1, alergia_bajo)
    alergia_media_1 = np.fmin(regla_2, alergia_medio)
    alergia_alta_1 = np.fmin(regla_3, alergia_alto)
    alergia_alta_2 = np.fmin(regla_4, alergia_alto)
    alergia_media_2 = np.fmin(regla_5, alergia_medio)
    alergia_baja_2 = np.fmin(regla_6, alergia_bajo)
    alergia_alta_3 = np.fmin(regla_7, alergia_alto)
    alergia_alta_4 = np.fmin(regla_8, alergia_alto)
    alergia_baja_3 = np.fmin(regla_9, alergia_bajo)
    alergia_baja_4 = np.fmin(regla_10, alergia_bajo)
    alergia_media_3 = np.fmin(regla_11, alergia_medio)
    alergia_media_4 = np.fmin(regla_12, alergia_medio)
    alergia_alta_5 = np.fmin(regla_13, alergia_alto)
    alergia_alta_6 = np.fmin(regla_14, alergia_alto)
    alergia_alta_7 = np.fmin(regla_15, alergia_alto)
    alergia_alta_8 = np.fmin(regla_16, alergia_alto)
    alergia_media_5 = np.fmin(regla_17, alergia_medio)

    # Visualización de las reglas
    plt.subplot(fila, columna, 5)
    plt.title('Activación de las reglas')
    plt.plot(x_alergia, alergia_baja_1, label='Regla 1', marker='o')
    plt.plot(x_alergia, alergia_media_1, label='Regla 2', marker='o')
    plt.plot(x_alergia, alergia_alta_1, label='Regla 3', marker='o')
    plt.plot(x_alergia, alergia_alta_2, label='Regla 4', marker='o')
    plt.plot(x_alergia, alergia_media_2, label='Regla 5', marker='v')
    plt.plot(x_alergia, alergia_baja_2, label='Regla 6', marker='v')
    plt.plot(x_alergia, alergia_alta_3, label='Regla 7', marker='v')
    plt.plot(x_alergia, alergia_alta_4, label='Regla 8', marker='v')
    plt.plot(x_alergia, alergia_baja_3, label='Regla 9', marker='^')
    plt.plot(x_alergia, alergia_baja_4, label='Regla 10', marker='^')
    plt.plot(x_alergia, alergia_media_3, label='Regla 11', marker='^')
    plt.plot(x_alergia, alergia_media_4, label='Regla 12', marker='^')
    plt.plot(x_alergia, alergia_alta_5, label='Regla 13', marker='s')
    plt.plot(x_alergia, alergia_alta_6, label='Regla 14', marker='s')
    plt.plot(x_alergia, alergia_alta_7, label='Regla 15', marker='s')
    plt.plot(x_alergia, alergia_alta_8, label='Regla 16', marker='s')
    plt.plot(x_alergia, alergia_media_5, label='Regla 17', marker='s')
    plt.title('Activación de las reglas')
    plt.xlabel('Nivel de alergia')
    plt.ylabel('Grado de pertenencia')
    plt.legend()

    # Agregación
    agregacion1 = np.fmax(alergia_baja_1, alergia_media_1)
    agregacion2 = np.fmax(agregacion1, alergia_alta_1)
    agregacion3 = np.fmax(agregacion2, alergia_alta_2)
    agregacion4 = np.fmax(agregacion3, alergia_media_2)
    agregacion5 = np.fmax(agregacion4, alergia_baja_2)
    agregacion6 = np.fmax(agregacion5, alergia_alta_3)
    agregacion7 = np.fmax(agregacion6, alergia_alta_4)
    agregacion8 = np.fmax(agregacion7, alergia_baja_3)
    agregacion9 = np.fmax(agregacion8, alergia_baja_4)
    agregacion10 = np.fmax(agregacion9, alergia_media_3)
    agregacion11 = np.fmax(agregacion10, alergia_media_4)
    agregacion12 = np.fmax(agregacion11, alergia_alta_5)
    agregacion13 = np.fmax(agregacion12, alergia_alta_6)
    agregacion14 = np.fmax(agregacion13, alergia_alta_7)
    agregacion15 = np.fmax(agregacion14, alergia_alta_8)
    agregacion16 = np.fmax(agregacion15, alergia_media_5)

    # Defuzzificación
    alergia = fuzz.defuzz(x_alergia, agregacion16, 'bisector')

    # Visualización de la agregación
    plt.subplot(fila, columna, 6)
    plt.plot(x_alergia, agregacion16, label='Agregación', marker='o')
    plt.plot(alergia, 0, 'ro', label='Valor desfuzzificado')
    plt.title('Agregación y desfuzzificacion')
    plt.xlabel('Nivel de alergia')
    plt.ylabel('Grado de pertenencia')
    plt.legend()

    # Fuzzificación de la alergia resultante
    resultado_bajo = fuzz.interp_membership(x_alergia, alergia_bajo, alergia)
    resultado_medio = fuzz.interp_membership(x_alergia, alergia_medio, alergia)
    resultado_alto = fuzz.interp_membership(x_alergia, alergia_alto, alergia)
    resultado = max(resultado_bajo, resultado_medio, resultado_alto)
    if(resultado == resultado_bajo):
        return "baja", figura
    elif(resultado == resultado_medio):
        return "media", figura
    else:
        return "alta", figura



def interfaz():
    sg.theme('BlueMono')
    font_title = ('Arial', 24)
    font = ('Arial', 12)

    elementos_interfaz = [
        [sg.Text("Alergia estacional", font = font_title, text_color = "#FFFFFF")],
        [sg.Text("Bienvenido a la aplicación que te ayudara a descubrir que tan alergico eres!")],
        [sg.Text("Para la identificación de tu grado de alergia responda las siguientes preguntas", justification= "center")],
        [sg.Text("Recuerde que para contestar a las preguntas, utilice valores desde 1 hasta 10", justification= "center")],
        [sg.Text("Dado a que la alergia estacional depende de la estación, responda en base a aquella estacion en la que mas sufra, las siguientes preguntas", justification= "center")],
        [sg.Text("Puede considerar valores decimales utilizando un punto como separación", font = ("Arial", 12, "bold"), justification= "center")],
        [sg.Text("En una escala de 1 a 10, ¿Cuánto es su nivel de estornudo? ", justification = "center")],
        [sg.InputText(key = "estornudo", size = (30, 1), font = font)],
        [sg.Text("En una escala de 1 a 10, ¿Que tanto le lloran los ojos en la estación escogida?", justification = "center")],
        [sg.InputText(key = "ojos", size = (30, 1), font = font)],
        [sg.Text("En una escala de 1 a 10, ¿Que tanta picazon se le produce al estar en espacios exteriores en la epoca escogida?", justification = "center")],
        [sg.InputText(key = "picazon", size = (30, 1), font = font)],
        [sg.Text("Su alergia es: ", font = ("Arial", 12, "bold"), key = "resultado")],
        [sg.Button("Confirmar"), sg.Button("Salir")]
        ]
    
    layout = [
        [sg.Column(elementos_interfaz)]
    ]

    window = sg.Window("AlergiaEstacional", layout, font = font)

    while True:
        evento, valores = window.read()
        if evento == sg.WIN_CLOSED or evento == "Salir":
            break
        elif evento == "Confirmar":
            try:
                estornudo = float(valores["estornudo"])
                ojos = float(valores["ojos"])
                if estornudo < 1 or estornudo > 10 or ojos < 1 or ojos > 10:
                    sg.popup("Por favor, ingrese valores del 1 al 10", title = "Error")
                else:
                    result = inferir_alergia(valores["ojos"], valores["estornudo"], valores["picazon"])
                    window.Element("resultado").Update("Su alergia es: " + str(result[0]))
                    fig = result[1]
                    plt.show()
            except ValueError:
                if valores["estornudo"] == "" or valores["ojos"] == "" or valores["picazon"] == "":
                    sg.popup("Por favor, responda a todos los campos", title = "Error")
                else:
                    sg.popup("Por favor, ingrese valores numéricos", title = "Error")
                continue
    window.close()
# Ejecutar el programa
interfaz()