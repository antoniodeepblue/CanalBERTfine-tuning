# Importar las bibliotecas necesarias
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
context = "Averías: En primer lugar, averigua si la falta de agua afecta a todos los puntos de agua de tu casa. Si no, pregunta a algún vecino si tiene agua. En caso de que sí tenga, comprueba que la llave de paso interior de tu vivienda esté completamente abierta y verifica que el contador o el equipo de medición tenga las dos llaves de paso abiertas (es decir, giradas completamente a la izquierda). En la batería de contadores hay un plano donde podrás encontrar la ubicación de tu contador. Si después de haber hecho estas comprobaciones sigues sin agua, ponte en contacto con nosotros. Comprueba que la llave de paso interior de tu casa esté completamente abierta. Si tienes instalado algún aparato reductor de presión, comprueba que funcione correctamente. A continuación, verifica que el contador o el equipo de medición tenga las dos llaves de paso abiertas (es decir, giradas completamente a la izquierda). Sí, hemos puesto en marcha un nuevo servicio para comunicarte de manera inmediata las incidencias que pudieran producirse en la red de distribución, bien por obras de mejora o por situaciones sobrevenidas, y la duración estimada de la suspensión temporal del suministro. Este servicio se ha desarrollado para que puedas adoptar las medidas necesarias para reducir el impacto que estas intervenciones pudieran ocasionar en tu actividad diaria. Ponte en contacto con nosotros a través del teléfono gratuito 900 365 365, o date de alta desde tu perfil (para ello deberás estar registrado). Durante la noche, cuando ya no se vaya a realizar consumo de agua, cierra todos los grifos y toma la lectura del contador. Al día siguiente, antes de abrir cualquier grifo, anota de nuevo la lectura. Si la lectura es diferente, es posible que exista una fuga en la instalación. Cierra todos los grifos: Comprueba que todos aquellos electrodomésticos que consumen agua no están en funcionamiento. Asegúrate también de que las llaves de paso anteriores y posteriores al contador están abiertas. Verifica que tu contador no registra consumo: Comprueba que las fracciones de metros cúbicos permanecen en la misma posición pasado un tiempo. Si observas un avance continuado de estos dígitos sin que se esté produciendo ningún consumo, es posible que exista una fuga. En tal caso, habrá que localizarla y repararla en el menor tiempo posible. En las zonas verdes, es recomendable conocer los caudales reales de riego de cada instalación. Para ello, puedes contabilizar el consumo registrado por el contador durante un tiempo determinado mientras tienes activado únicamente el riego. Este caudal te servirá de referencia para periódicamente verificar su valor. En caso de que, exclusivamente durante el riego, detectes un incremento injustificado de ese valor normal, será un indicador de la aparición de una fuga en la instalación de riego. En primer lugar, averigua si la falta de agua afecta a todos los puntos de agua de tu casa. Si no, pregunta a algún vecino si tiene agua. En caso de que sí tenga, comprueba que la llave de paso interior de tu vivienda esté completamente abierta y verifica que el contador o el equipo de medición tenga las dos llaves de paso abiertas (es decir, giradas completamente a la izquierda). En la batería de contadores hay un plano donde podrás encontrar la ubicación de tu contador. Si después de haber hecho estas comprobaciones sigues sin agua, ponte en contacto con nosotros. En numerosas ocasiones, las fugas interiores se deben a pequeños goteos en grifos, calderas o cisternas, por lo que es conveniente controlar periódicamente el consumo de estos. Para un mejor mantenimiento de tu instalación interior: 1.Observa que la cisterna del inodoro no tiene pérdidas: Hazlo una vez haya terminado el llenado de la cisterna. Además de la revisión visual y sonora, puedes añadir colorante alimentario en el tanque para que te resulte más fácil detectar una posible fuga. 2.Comprueba el estado de grifos y calderas: Vigila que no goteen. Verifica también que el circuito cerrado de la calefacción o la caldera funciona correctamente. A veces se producen fallos que hacen que el agua se pierda directamente por la red de saneamiento. Es recomendable que conozcas el trazado de las tuberías, así como la localización de las llaves de corte o seccionamiento. Accionando estas llaves se puede acotar el tramo de la instalación en el que se localiza la fuga para así facilitar las labores de reparación. Contacto, gestiones y contratos: Puedes elegir siete vías diferentes para contactar con nosotros: La manera más rápida es utilizar el chat de esta Oficina Virtual. Si prefieres hablar por teléfono, llama a nuestro teléfono gratuito 900 365 365 en horario de 08:00 a 20:00. Si necesitas comunicar una incidencia o avería, el servicio se encuentra operativo las 24 horas del día, los 365 días del año. Otra opción telefónica es que te llamemos nosotros. Solo tienes que indicar tu número de teléfono y el rango horario que prefieras. Puedes utilizar un formulario de contacto. Tienes la posibilidad de escribirnos un correo electrónico a la dirección: clientes@canaldeisabelsegunda.es. Puedes pedir cita para que te atendamos en nuestros centros de atención al cliente. Y si prefieres métodos tradicionales, puedes enviar una comunicación escrita por correo ordinario o fax: Att. Registro General. C/ Santa Engracia, 125 (28003 - Madrid). Horario: de lunes a viernes de 8:30 a 14:00 horas. Fax: 915 451 430. Independientemente del canal de comunicación que prefieras utilizar, no olvides indicarnos como referencia tu número de contrato o la dirección de la finca objeto de la solicitud, así como un teléfono de contacto. Si no eres usuario de la Oficina virtual puedes realizar las siguientes gestiones: 1.Relacionadas con contratación: Contratar un suministro, darte de alta como cliente sensible y realizar gestiones asociadas con alcantarillado y saneamiento. 2.Relacionadas con tus facturas: Pagar tus facturas, visualizar facturas electrónicas, realizar simulaciones de facturas, solicitar bonificaciones y darte de alta en la factura electrónica. 3.Relativas a consumos y lecturas: Consultar la fecha de la próxima lectura y enviar la lectura de tu contador. 4.Otro tipo de gestiones: Cambiar la titularidad de un contrato, darte de alta en la Oficina virtual, pedir una cita para atención personalizada en una oficina, gestionar tus citas, comunicar incidencias, consultar solicitudes, acceder a la oficina de reclamaciones y notificar incidencias en el suministro. Si ya estás dado de alta en la Oficina Virtual, además de todas las gestiones recogidas en el apartado anterior, puedes acceder a las siguientes: 1.Trámites que afectan a tus contratos: Actualización de los datos de contacto y modificación de los datos de pago. 2.Asociados a tus facturas: Consultar facturas anteriores, reclamar alguna factura, enviarnos un justificante de pago, solicitar una devolución de saldo y descargar un listado de tus facturas. 3.Relacionadas con cancelaciones: Dar de baja un suministro y darte de baja como usuario de la Oficina virtual. 4.Otras gestiones: Consultar el estado de tus quejas y acceder al arbitraje de consumo. Facturación: En tu factura podrás distinguir cuatro conceptos facturados: aducción, distribución, depuración y alcantarillado. Cada uno de ellos se ve afectado por unas ratios diferentes y consta de dos partes. Por un lado, una cuota de servicio que es fija y se factura independientemente de que exista o no consumo. Por otro, una parte variable que dependerá del consumo de agua realizado en el periodo que se factura. La factura es enviada bimestralmente. Es decir, cada dos meses. Es decir, cada dos meses. Una vez emitida la factura, tienes un periodo voluntario de pago de 30 días naturales. El Reglamento para el Servicio y Distribución de las Aguas de Canal de Isabel II (Decreto 2922/1975, de 31 de octubre) establece varios métodos para facturar el consumo de agua: 1.Estimación de consumos (EST): este método se utiliza cuando no es posible acceder al contador en la fecha en que debe realizarse la lectura. Su cálculo se efectúa de acuerdo con el consumo medio diario de los dos periodos análogos precedentes y, en caso de disponerse solo del histórico de consumos de un año, la estimación se realizará de acuerdo con el consumo medio diario del periodo análogo precedente. El consumo facturado de esta forma se considera a cuenta. Por tanto, una vez que se tome la lectura real del contador, los metros cúbicos facturados de esta forma se descontarán del consumo realizado. 2.Evaluación de consumos (EV): para facturar los metros cúbicos mediante este método, se debe dar la circunstancia de que, aun teniendo acceso al contador, no se pueda recoger la lectura porque exista una anomalía que requiera su sustitución. El cálculo del consumo se realiza de forma análoga a la estimación. 3.Diferencia de lecturas del aparato de medida (DFI): diferencia de la lectura tomada en el periodo anterior y la actual. Puedes consultar las tarifas aplicables para la facturación del servicio de suministro de agua en la sección de tarifas de esta web.  Estudiamos individualmente las solicitudes, y tras su análisis, concedemos, de acuerdo con una serie de criterios, el número de plazos e interés correspondiente según el importe y las características del cliente. Según establece el Reglamento para el Servicio y Distribución de las Aguas de Canal de Isabel II, (Decreto 2922/1975, de 31 de octubre) podríamos suspender el suministro de agua si transcurridos 30 días naturales desde la emisión de la factura, esta no ha sido abonada. El corte del suministro implicaría continuar facturando las cuotas de servicio. El restablecimiento del servicio se realizará una vez liquidada la deuda, así como el importe del restablecimiento. Transcurridos tres meses sin que se haya producido el pago, Canal puede resolver el contrato y proceder a la condena de la acometida (retirada de la instalación). Para volver a tener agua habrá que realizar una nueva contratación. Bonificaciones: Tal y como se establece en las Órdenes 1330/2018, de 18 de abril y 2586/2018, de 18 de diciembre, existen tres tipos de bonificaciones por las que nuestros clientes pueden verse beneficiados: por pensión de viudedad, por exención social o por familia o vivienda numerosa. Comprometidos con las personas. En Canal de Isabel II llevamos años adaptándonos a las necesidades especiales de nuestros clientes. Nuestra tarifa social de consumo de agua está pensada para quienes más lo necesitan y contempla distintas bonificaciones: por exención social, por familia o vivienda numerosa y también por pensión de viudedad, así como en viviendas en situación de ocupación ilegal. Puede beneficiarse de las bonificaciones del Canal de Isabel II: Perceptores de una pensión por viudedad. Familias o viviendas numerosas. Entidades sin ánimo de lucro titulares de viviendas comunitarias o pisos tutelados. Beneficiarios de una pensión no contributiva. Beneficiarios de la renta mínima de inserción Beneficiarios de la renta activa de inserción. Perceptores del ingreso mínimo vital. Personas en situación de especial exclusión que cuenten con un informe del trabajador social. Personas con vivienda legítima en situación de ocupación ilegal. Pensión de viudedad.  Perceptores de una pensión por viudedad con renta total inferior a 14.000 euros anuales. La bonificación será del 50 % del importe de la cuota de servicio fija. Documentación que debes aportar por pensión de viudedad: Certificado, emitido por los órganos o entidades competentes, que acredite la condición de perceptor de una pensión de viudedad y el importe de la misma. Fotocopia del Documento Nacional de Identidad del solicitante. Solicitud de bonificación de la factura de consumo de agua, marcando la casilla de declaración responsable de que sus ingresos totales, incluida la pensión de viudedad, no superan el importe establecido. Exención social del Canal de Isabel II. Aquellos usuarios que acrediten no poder hacer frente al pago de la factura en su vivienda habitual. También las entidades sin ánimo de lucro que sean titulares de viviendas comunitarias o pisos tutelados. La bonificación será del 50 % del importe de la cuota de servicio fija. Además, se bonificará el importe total de la parte variable del consumo realizado hasta 25 m3 /bimestre. Documentación que debes aportar por exención social: Certificado que acredite la condición de beneficiario de la renta mínima de inserción. Certificado que acredite la condición de beneficiario de una pensión no contributiva. Certificado que acredite la condición de beneficiario de la renta activa de inserción. Certificado que acredite la condición de beneficiario del ingreso mínimo vital. Informe del trabajador social que valore favorablemente la concesión de la bonificación por exención social. Familia o vivienda numerosa que se bonifican en el Canal de Isabel II. Usuarios que acrediten que su vivienda habitual está habitada por una familia numerosa o por más de 4 personas empadronadas. Las familias numerosas de categoría general y las viviendas de 5 a 7 personas pagarán el consumo realizado en el segundo bloque tarifario a precios del primero y contarán con una reducción del 10 % en el importe de la parte variable. Las familias numerosas de categoría especial y las viviendas de más de 7 personas contarán con la bonificación indicada anteriormente y, además, pagarán a precios del segundo bloque tarifario hasta un máximo de 30 m3 /bimestre del consumo correspondiente al tercer bloque. Documentación que debes aportar por familia o vivienda numerosa: Familia numerosa: título en vigor de familia numerosa. Vivienda numerosa: certificado de empadronamiento. Ocupación ilegal. Las personas físicas titulares del contrato con Canal de Isabel II de suministro destinado a uso doméstico o, en su caso, las personas físicas usuarias que sean poseedores legítimos de una vivienda, y que no puedan disponer de ella por haber sido privados de la misma, siempre que acrediten documentalmente de forma fehaciente, la iniciación de un procedimiento por allanamiento o usurpación de vivienda, o la iniciación de un procedimiento para la recuperación posesoria frente al particular que, sin habitar en ella y careciendo de título anterior o actual, entrara u ocupara la vivienda sin su consentimiento y contra su voluntad. Se bonificará el 100 % de la parte fija y variable de la factura durante el tiempo que dure la ocupación. Inicialmente tendrá una vigencia de 6 meses que podrá prorrogarse. Documentación que debes aportar por ocupación ilegal: Impreso de solicitud de bonificación de factura de consumo de agua. Documentación fehaciente de iniciación de un procedimiento por allanamiento o usurpación de vivienda, o la iniciación de un procedimiento para la recuperación posesoria frente al particular que, sin habitar en ella y careciendo de título anterior o actual, entrara u ocupara la vivienda sin su consentimiento y contra su voluntad. Certificado de empadronamiento en la vivienda en relación con la que se solicita la aplicación de la bonificación cuando el usuario no sea el titular del contrato suministro de agua o, en su defecto, acreditación por cualquiera de los medios de prueba generalmente admitidos en derecho, del justo título por el que tiene la condición de usuario y está obligado al pago del suministro. Consumo de agua correspondiente a la vivienda ocupada, certificado por el órgano de la comunidad de propietarios competente para ello. (sólo en suministros a pluriviviendas). Declaración responsable de que subsisten las condiciones para ser beneficiario de la bonificación y de que no se ha podido recuperar la posesión de la vivienda. (para solicitudes de prórrogas). Entidades sin ánimo de lucro. Documentación que deben aportar las entidades sin ánimo de lucro: Acreditación de estar inscritas en el Registro de Entidades, Centros y Servicios de Acción Social de la Consejería de Políticas Sociales y Familia. Consumos y lecturas: El número que aparece en la pantalla del contador sin tener en cuenta las fracciones de metro cúbico que normalmente aparecen en otro tamaño o color. Puedes enviar la lectura del contador a Canal de Isabel II: 1. A través de la Oficina Virtual, en la sección Gestiones online. 2. A través de nuestro teléfono gratuito de atención al cliente: 900 365 365. 3. En cualquiera de nuestras oficinas de atención al cliente. 4. Por correo electrónico: clientes@canaldeisabelsegunda.es. Formas de pago: La manera más cómoda para realizar el pago de tus facturas es mediante domiciliación bancaria. Si lo deseas, puedes actualizar los datos de tu cuenta desde tu perfil o llamando a nuestro teléfono gratuito 900 365 365 en horario de 08:00 a 20:00 todos los días laborables. Otra opción telefónica es que te llamemos nosotros (solo tienes que indicar tu número de teléfono y el rango horario que prefieras). También tienes la posibilidad de abonar tus facturas con tarjeta de crédito. Para realizar el pago dispones de dos canales: en esta Oficina virtual o a través de nuestro teléfono gratuito 900 365 365 en horario de 08:00 a 20:00 todos los días laborables; también podemos llamarte nosotros si lo deseas (solo tienes que indicar tu número de teléfono y el rango horario que prefieras). Otra opción es acudir a una entidad bancaria presentando la factura. También puedes pagar las facturas en nuestros centros de atención al cliente. Contratación y cambio de titularidad: La gestión de contratar un nuevo suministro la puedes hacer de varias maneras: a través del apartado correspondiente de esta Oficina Virtual; por teléfono, llamando al número gratuito 900 365 365 de 08:00 a 20:00 o pidiendo que te llamamos nosotros (solo tienes que indicar tu número de teléfono y el rango horario que prefieras); o presencialmente en nuestro centro de atención al cliente. Para contratar un suministro, si eres un particular, te hará falta tener a mano tu documento identificativo (DNI, Tarjeta de residencia, etc.). En el caso de que el titular sea una empresa o un administrador de fincas, necesitas el DNI y el documento acreditativo del apoderado, administrador o responsable de la empresa o finca (en formato PDF o imagen). Con carácter general, para realizar la puesta en servicio del suministro en un contador secundario en batería (viviendas en altura), el plazo será de 3 días laborales desde la fecha de la contratación. Para nuevas acometidas que impliquen nuevo contrato, siempre que exista red de distribución adecuada a la que conectarla, el plazo será de 10 días naturales desde la fecha en que Canal de Isabel II reciba los permisos y licencias de los organismos correspondiente para poder realizarla. El permiso de ocupación temporal de calzada se solicita, siempre que sea necesario, para poder llevar a cabo las actuaciones de puesta en marcha del suministro. Se distingue: 1. Permiso de corte de tráfico: implica cortar la circulación. 2. Permiso de ocupación de calzada: no implica un corte de la circulación, sin embargo, se precisa para poder dejar el material necesario para la ejecución de la obra. Este permiso autoriza la ocupación de la vía pública en calzada y en los casos necesarios de la zona de aparcamiento. Estos permisos son requeridos dependiendo de la situación de la finca a suministrar. El Decreto 3068/75, de 31 de octubre, establece que en la contratación del suministro se pueden facturar los siguientes conceptos: 1. Cuota de red: corresponde al importe de las obras de adaptación y mejora de la red existente. Este concepto se facturará como norma general para los nuevos suministros solicitados. 2. Cuota de enganche: se trata de los costes derivados de la ejecución de la obra, tales como materiales y mano de obra. 3. Anticipo de consumo: es una cantidad que cubre el importe de la facturación del suministro de agua y que responde a las obligaciones económicas contraídas. Este importe será devuelto al cliente a la finalización de la relación contractual, mediante la solicitud expresa del mismo y siempre y cuando se hayan cumplido las obligaciones contraídas por parte de este. Este concepto no existe en el caso de acometidas de protección contra incendios (PCI) y en los distribuidores principales (DP) y distribuidores principales únicos (DPU) que dan servicio a baterías de contadores. Si quieres contratar el suministro en una ubicación en la que ya existe un contrato activo, el proceso de cambio de titularidad implica la cancelación del contrato antiguo y el alta de uno nuevo. Si eres un particular, te hará falta tener a mano tu documento identificativo (DNI, Tarjeta de residencia, etc.). En el caso de que el titular sea una empresa o comunidad de propietarios, necesitas el DNI y el documento acreditativo del apoderado, administrador o responsable de la empresa o finca (en formato PDF o imagen). El coste del cambio de titularidad corresponde al importe del anticipo de consumo. Hay que presentar la licencia de primera ocupación para contratar la acometida definitiva porque así viene recogido en el artículo 160 de la Ley 9/2001 del Suelo de la Comunidad de Madrid. Este artículo establece que las empresas suministradoras de agua exigirán, para la contratación definitiva de los suministros o servicios respectivos en edificio o construcciones de nueva planta o ampliaciones, la acreditación de la licencia municipal de la primera ocupación. La vigencia de un contrato provisional se establece en base a la licencia de obra porque así viene recogido en el artículo 160 de la Ley 9/2001 del Suelo de la Comunidad de Madrid. El artículo 160 de la Ley 9/2001 del Suelo de la Comunidad de Madrid establece que las empresas suministradoras de agua exigirán para la contratación provisional de los respectivos servicios la acreditación de la licencia municipal de obra. La ubicación del armario y del contador o aparato de medida debe encontrase al nivel de la vía pública porque así se establece en el Decreto 2922/1975, de 31 de octubre. Esta ubicación es necesaria ya que, de este modo, nuestro personal puede tomar la lectura del contador, notificar las posibles fugas según la lectura tomada, así como reparar cualquier avería producida en la acometida y el conjunto de medida. Por otro lado, la Orden 2106/1994, de 11 de noviembre, establece que los contadores cuyo calibre sea igual o inferior a 65 milímetros deberán estar alojados en un armario ubicado en el muro de fachada o en cerramiento, si este último existiera. Para los contadores cuyo calibre supere los 65 milímetros, existe la posibilidad de alojarlos en hornacina en las mismas condiciones que en el caso anterior. Cuando una finca cambia de propietario y existe contrato de suministro de agua hay obligación de realizar el cambio de titularidad porque lo establece el Decreto 2922/1975, de 31 de octubre. El incumplimiento de este precepto tendrá la consideración de infracción grave y podrá incluso sancionarse con la facturación de un recargo de hasta 2.000 m3 de agua, valorados según tarifa general. Para realizar la baja del suministro se solicita a través de esta Oficina Virtual, donde también puedes consultar los requisitos necesarios para efectuar dicha gestión. Esta gestión tiene un coste que varía en función de la obra a realizar."

# Función para responder preguntas dadas una pregunta y un contexto predefinido
def answer_question(question):
    # Obtener la respuesta del modelo
    result = nlp(question=question, context=context)

    # Verificar si la respuesta supera un cierto score de confianza
    if result["score"] > 0.3:
       # Verificar si se encontró una respuesta
       if result['answer']:  # Si la respuesta no está vacía
           # Retornar la respuesta encontrada
           return result['answer']
       else:
           # Devolver un mensaje indicando que la pregunta debe ser reformulada
           return "Lo siento, no pude encontrar una respuesta para tu pregunta. Por favor, reformula tu pregunta."
    else:
       # Devolver un mensaje indicando que la pregunta debe ser reformulada
       return "Lo siento, no estoy seguro de la respuesta. Por favor, reformula tu pregunta."

# Definir la interfaz Gradio
iface = gr.Interface(fn=answer_question,
                     inputs= gr.Textbox(label="Question", placeholder="Escribe tu pregunta aquí...", scale=7),
                     outputs=gr.Textbox(label="Answer"),
                     theme="soft",
                     title="BERT_fine-tuning: Preguntas y Respuestas del Canal de Isabel II",
                     description='Autor: <a href=\"https://huggingface.co/Antonio49\">Antonio Fernández</a> de <a href=\"https://www.canaldeisabelsegunda.es/\">Canal de Isabel II</a>. Formación: <a href=\"https://www.uoc.edu/es/\">Grado Ingeniería Informática</a> Aplicación desarrollada para TFG_Inteligencia_Artificial.'"   Proporcione una pregunta y el asistente encontrará la respuesta.")

if __name__ == "__main__":
    # Lanzar la interfaz
    iface.launch()
