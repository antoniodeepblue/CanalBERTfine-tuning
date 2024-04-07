import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import gradio as gr

# Cargar el modelo y el tokenizador
model_name ='Antonio49/Personal'
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configurar el pipeline para la pregunta-respuesta
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Contexto integrado en el código
context = "En primer lugar, averigua si la falta de agua afecta a todos los puntos de agua de tu casa. Si no, pregunta a algún vecino si tiene agua. En caso de que sí tenga, comprueba que la llave de paso interior de tu vivienda esté completamente abierta y verifica que el contador o el equipo de medición tenga las dos llaves de paso abiertas (es decir, giradas completamente a la izquierda). En la batería de contadores hay un plano donde podrás encontrar la ubicación de tu contador. Si después de haber hecho estas comprobaciones sigues sin agua, ponte en contacto con nosotros. Comprueba que la llave de paso interior de tu casa esté completamente abierta. Si tienes instalado algún aparato reductor de presión, comprueba que funcione correctamente. A continuación, verifica que el contador o el equipo de medición tenga las dos llaves de paso abiertas (es decir, giradas completamente a la izquierda). Sí, hemos puesto en marcha un nuevo servicio para comunicarte de manera inmediata las incidencias que pudieran producirse en la red de distribución, bien por obras de mejora o por situaciones sobrevenidas, y la duración estimada de la suspensión temporal del suministro. Este servicio se ha desarrollado para que puedas adoptar las medidas necesarias para reducir el impacto que estas intervenciones pudieran ocasionar en tu actividad diaria. Ponte en contacto con nosotros a través del teléfono gratuito 900 365 365, o date de alta desde tu perfil (para ello deberás estar registrado). Durante la noche, cuando ya no se vaya a realizar consumo de agua, cierra todos los grifos y toma la lectura del contador. Al día siguiente, antes de abrir cualquier grifo, anota de nuevo la lectura. Si la lectura es diferente, es posible que exista una fuga en la instalación. Cierra todos los grifos: Comprueba que todos aquellos electrodomésticos que consumen agua no están en funcionamiento. Asegúrate también de que las llaves de paso anteriores y posteriores al contador están abiertas. Verifica que tu contador no registra consumo: Comprueba que las fracciones de metros cúbicos permanecen en la misma posición pasado un tiempo. Si observas un avance continuado de estos dígitos sin que se esté produciendo ningún consumo, es posible que exista una fuga. En tal caso, habrá que localizarla y repararla en el menor tiempo posible. En las zonas verdes, es recomendable conocer los caudales reales de riego de cada instalación. Para ello, puedes contabilizar el consumo registrado por el contador durante un tiempo determinado mientras tienes activado únicamente el riego. Este caudal te servirá de referencia para periódicamente verificar su valor. En caso de que, exclusivamente durante el riego, detectes un incremento injustificado de ese valor normal, será un indicador de la aparición de una fuga en la instalación de riego. En primer lugar, averigua si la falta de agua afecta a todos los puntos de agua de tu casa. Si no, pregunta a algún vecino si tiene agua. En caso de que sí tenga, comprueba que la llave de paso interior de tu vivienda esté completamente abierta y verifica que el contador o el equipo de medición tenga las dos llaves de paso abiertas (es decir, giradas completamente a la izquierda). En la batería de contadores hay un plano donde podrás encontrar la ubicación de tu contador. Si después de haber hecho estas comprobaciones sigues sin agua, ponte en contacto con nosotros. En numerosas ocasiones, las fugas interiores se deben a pequeños goteos en grifos, calderas o cisternas, por lo que es conveniente controlar periódicamente el consumo de estos. Para un mejor mantenimiento de tu instalación interior: 1.Observa que la cisterna del inodoro no tiene pérdidas: Hazlo una vez haya terminado el llenado de la cisterna. Además de la revisión visual y sonora, puedes añadir colorante alimentario en el tanque para que te resulte más fácil detectar una posible fuga. 2.Comprueba el estado de grifos y calderas: Vigila que no goteen. Verifica también que el circuito cerrado de la calefacción o la caldera funciona correctamente. A veces se producen fallos que hacen que el agua se pierda directamente por la red de saneamiento. Es recomendable que conozcas el trazado de las tuberías, así como la localización de las llaves de corte o seccionamiento. Accionando estas llaves se puede acotar el tramo de la instalación en el que se localiza la fuga para así facilitar las labores de reparación."

def answer_question(question):
    """
    Función para responder preguntas dadas una pregunta y un contexto predefinido.
    """
    # Obtener la respuesta del modelo
    result = nlp(question=question, context=context)
    # Retornar la respuesta encontrada
    return result['answer']

# Definir la interfaz Gradio
iface = gr.Interface(fn=answer_question,
                     inputs= gr.Textbox(label="Question", placeholder="Escribe tu pregunta aquí...", scale=7),
                     outputs=gr.Textbox(label="Answer"),
                     theme="soft",
                     title="Preguntas y Respuestas BERTfinetuning con contexto predefinido",
                     description='Autor: <a href=\"https://huggingface.co/Antonio49\">Antonio Fernández</a> de <a href=\"https://www.canaldeisabelsegunda.es/\">Canal de Isabel II</a>. Formación: <a href=\"https://www.uoc.edu/es/\">Grado Ingeniería Informática</a> Aplicación desarrollada para TFG'"   Proporcione una pregunta y el asistente encontrará la respuesta.")

if __name__ == "__main__":
    # Lanzar la interfaz
    iface.launch()
