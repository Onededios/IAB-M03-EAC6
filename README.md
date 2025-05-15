# IAB-M03-EAC6 - Joel Olivera

## Estructura del proyecto

El proyecto se estructura mediante archivos de python, archivos ejecutables (bat) y archivos de texto o configuración.

El código "main.py" es el principal archivo del código, en el que se llaman a las funciones del archivo "functions.py".
Una vez ejecutado el código "main.py" mediante python, se generará una carpeta img, que contendrá las imágenes generadas respecto a la asignación de clústeres del modelo KMeans.

El archivo "test_functions.py" tiene declarados unit tests para las funciones del archivo "functions.py".

El archivo "gen_docs.bat" genera un archivo de documentación "docs/functions" a partir de la docstring de las funciones del archivo "functions.py"·

El archivo "bootstrap.bat" es el encargado de instalar las dependencias para ejecutar el script.

El archivo "run_lint.bat" es el encargado de generar un archivo "lint_result.txt" con el resultado de clean code según las normas definidas en "pylintrc".

## Instalación del proyecto

Para instalar el proyectyo hay que ejecutar el archivo "bootstrap.bat" o ejecutar el comando "pip install -r requirements.txt".

## Ejecución del proyecto

Para ejecutar el proyecto simplemente hay que tener las dependencias del mismo instaladas y ejecutar el mismo con python mediante "py main.py".

## Comprobación del análisis estático

Para ejecutar el proceso de análisis estático del proyecto, hay que ejecutar el archivo "run_lint.bat".

## Generación de documentación

Para generar la documentación del archivo "functions.py" hay que ejecutar el archivo "gen_docs.bat".

## Comprobación de los tests

Para comprobar que las funciones del archivo "functions.py" empleadas en el archivo "main.py" siguen funcionando correctamente, hay que ejecutar el archivo "test_functions.py" mediante "py test_functions.py".
