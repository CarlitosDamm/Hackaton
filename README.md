# Clasificador de Calidad de Vinos con MLOps
## API de Predicción con FastAPI, Docker, MLflow y Monitoreo

- Carlos Alberto Damm Manzanera

**Materia:** Gestión de Proyectos de Inteligencia Artificial

**Universidad:** Tecmilenio

---

# Descripción del proyecto

Este proyecto implementa una solución completa de **Machine Learning Operations (MLOps)** para la clasificación automática de calidad de vinos utilizando técnicas modernas de despliegue, monitoreo y observabilidad.

El sistema parte de un modelo de **Machine Learning** entrenado con el **Wine Quality Dataset** de la UCI Machine Learning Repository y posteriormente lo convierte en un servicio de inferencia listo para producción mediante una API REST desarrollada con **FastAPI**.

Además del despliegue del modelo, el proyecto incorpora múltiples prácticas de MLOps:

- API REST para inferencias en tiempo real.
- Contenerización mediante Docker.
- Orquestación con Docker Compose.
- Frontend web para consumo del servicio.
- Monitoreo de métricas del sistema.
- Detección automática de Data Drift.
- Registro de inferencias utilizando MLflow.
- Healthchecks entre servicios para garantizar un inicio seguro.

Todo el entorno puede desplegarse mediante un único comando.

---

# Tabla de contenido

1. Objetivos
2. Arquitectura del sistema
3. Tecnologías utilizadas
4. Dataset
5. Modelo de Machine Learning
6. API REST
7. Monitoreo y observabilidad
8. Integración con MLflow
9. Detección de Data Drift
10. Docker y Docker Compose
11. Instalación
12. Uso de la API
13. Capturas del sistema
14. Resultados
15. Mejoras futuras
16. Créditos

---

# 1. Objetivos

El objetivo principal del proyecto consiste en demostrar un flujo completo de **Machine Learning Operations (MLOps)** que permita llevar un modelo entrenado hasta un entorno similar a producción.

Los objetivos específicos son:

- Entrenar un modelo de clasificación para calidad de vinos.
- Publicar el modelo mediante una API REST.
- Contenerizar todos los servicios utilizando Docker.
- Automatizar el despliegue mediante Docker Compose.
- Registrar todas las inferencias utilizando MLflow.
- Monitorear el comportamiento del sistema.
- Detectar automáticamente posibles desviaciones en los datos de entrada (Data Drift).
- Proporcionar una interfaz web sencilla para consumir el modelo.

---

# 2. Arquitectura del sistema

El proyecto está compuesto por tres servicios principales.

```

                Docker Compose

        ┌──────────────────────────┐
        │        Frontend          │
        │     HTML • CSS • JS      │
        │         Nginx            │
        └────────────┬─────────────┘
                     │ HTTP
                     ▼
        ┌──────────────────────────┐
        │         FastAPI          │
        │     Random Forest API    │
        └──────────┬───────────────┘
                   │
        ┌──────────┴───────────────┐
        │                          │
        ▼                          ▼

  Data Drift                 MLflow Tracking

```

### Flujo del sistema

1. El usuario realiza una solicitud desde el Frontend.
2. FastAPI valida los datos recibidos.
3. El modelo Random Forest genera una predicción.
4. Se registran métricas de desempeño.
5. Se analiza el Data Drift.
6. Se almacena la inferencia en MLflow.
7. La respuesta es enviada al usuario.

---

# 3. Tecnologías utilizadas

| Tecnología | Propósito |
|------------|-----------|
| Python 3.11 | Lenguaje principal |
| FastAPI | API REST |
| Docker | Contenerización |
| Docker Compose | Orquestación |
| MLflow | Tracking de inferencias |
| Nginx | Servidor web |
| Scikit-Learn | Modelo de Machine Learning |
| Pandas | Manipulación de datos |
| NumPy | Operaciones numéricas |
| Joblib | Persistencia del modelo |
| Pydantic | Validación de datos |

---

# 4. Dataset

Se utilizó el **Wine Quality Dataset** publicado por la **UCI Machine Learning Repository**, combinando los conjuntos de vino tinto y vino blanco.

Las características utilizadas incluyen:

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol
- Wine Type

La variable objetivo corresponde a una clasificación binaria:

| Clase | Significado |
|--------|-------------|
| 0 | Calidad estándar |
| 1 | Alta calidad |

---

# 5. Modelo de Machine Learning

Durante la etapa experimental se evaluaron tres algoritmos:

- Logistic Regression
- Decision Tree
- Random Forest

Tras la comparación de métricas y validación cruzada, el modelo seleccionado fue **Random Forest**, al obtener el mejor equilibrio entre precisión, recall, F1-Score y ROC-AUC.

Posteriormente el modelo fue serializado utilizando Joblib para ser consumido desde la API.

---

# 6. API REST

La aplicación utiliza FastAPI para exponer el modelo mediante una API REST.

## Endpoints

### Health Check

```
GET /health
```

Permite verificar que:

- API disponible
- Modelo cargado correctamente

---

### Predicción

```
POST /predict
```

Recibe las características fisicoquímicas del vino y devuelve:

- Predicción
- Probabilidad
- Tiempo de inferencia

---

### Métricas

```
GET /metrics
```

Devuelve información como:

- Número de solicitudes
- Errores
- Latencia promedio
- Distribución de predicciones

---

### Data Drift

```
GET /drift
```

Permite consultar:

- Número de solicitudes analizadas
- Variables fuera de distribución
- Estadísticas del detector de Drift

---

# 7. Monitoreo y observabilidad

Uno de los objetivos principales del proyecto consiste en monitorear continuamente el comportamiento del sistema.

Actualmente se registran:

- Tiempo promedio de respuesta
- Número de solicitudes
- Número de errores
- Distribución de clases predichas
- Latencia promedio
- Solicitudes procesadas
- Eventos relevantes mediante logs estructurados

Esto permite conocer el estado del sistema en tiempo real.

---

# 8. Integración con MLflow

Todas las inferencias realizadas por la API son registradas automáticamente en MLflow.

Entre la información almacenada se encuentran:

- Predicción realizada
- Tiempo de respuesta
- Parámetros utilizados
- Experimento asociado
- Métricas registradas

MLflow se ejecuta como un servicio independiente dentro de Docker Compose, permitiendo consultar el historial completo desde una interfaz web.

```
http://localhost:5000
```

---

# 9. Detección de Data Drift

El sistema implementa un detector básico de **Data Drift** utilizando estadísticas calculadas durante el entrenamiento.

Cada nueva solicitud es comparada contra la distribución original del dataset.

Cuando una característica supera el límite permitido, se registra un evento indicando:

- Variable afectada
- Valor recibido
- Media de entrenamiento
- Desviación estándar
- Límites aceptables

Esto permite detectar cambios en los datos de entrada que podrían afectar el desempeño del modelo.

---

# 10. Docker y Docker Compose

Todo el proyecto se ejecuta utilizando Docker Compose.

Los servicios desplegados son:

| Servicio | Descripción |
|----------|-------------|
| Frontend | Interfaz web |
| Backend | API FastAPI |
| MLflow | Tracking de inferencias |

MLflow utiliza volúmenes persistentes para conservar los experimentos incluso después de detener los contenedores.

Además, se implementó un **Healthcheck** que garantiza que MLflow esté completamente disponible antes de iniciar la API.

Esta práctica evita errores de inicialización durante el despliegue.

---

# 11. Instalación

## Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
```

Entrar al proyecto

```bash
cd Hackaton
```

Levantar todos los servicios

```bash
docker compose up --build
```

Una vez finalizado el despliegue estarán disponibles:

| Servicio | URL |
|----------|-----|
| Frontend | http://localhost:8080 |
| Swagger | http://localhost:8000/docs |
| Health | http://localhost:8000/health |
| Metrics | http://localhost:8000/metrics |
| Drift | http://localhost:8000/drift |
| MLflow | http://localhost:5000 |

---

# 12. Uso de la API

Ejemplo de solicitud:

```json
{
  "fixed acidity": 7.4,
  "volatile acidity": 0.70,
  "citric acid": 0.00,
  "residual sugar": 1.9,
  "chlorides": 0.076,
  "free sulfur dioxide": 11,
  "total sulfur dioxide": 34,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4,
  "wine_type": 0
}
```

Respuesta:

```json
{
  "prediction": 0,
  "probability": 0.91
}
```

---

# 13. Capturas del sistema

Se recomienda incluir las siguientes capturas:

- Página principal del Frontend.
- Swagger UI.
- Endpoint de métricas.
- Endpoint de Data Drift.
- Dashboard de MLflow.
- Registro de experimentos.
- Logs del sistema.

---

# 14. Resultados

Durante las pruebas realizadas se verificó correctamente:

- Despliegue completo mediante Docker Compose.
- Comunicación entre Frontend y Backend.
- Registro automático de inferencias.
- Monitoreo de métricas.
- Detección de Data Drift.
- Persistencia de experimentos mediante MLflow.
- Funcionamiento del Healthcheck.
- Inicialización automática de todos los servicios.

El sistema quedó completamente automatizado y puede iniciarse mediante un único comando.

---

# 15. Mejoras futuras

Como trabajo futuro podrían incorporarse:

- Prometheus para recolección de métricas.
- Grafana para dashboards en tiempo real.
- Registro de modelos mediante MLflow Model Registry.
- Integración continua (CI/CD).
- Pruebas automatizadas.
- Despliegue en Kubernetes.
- Autenticación de usuarios.
- Versionado automático de modelos.

---

# 16. Créditos

Proyecto desarrollado para la materia **Gestión de Proyectos de Inteligencia Artificial**.

- Carlos Alberto Damm Manzanera

**Universidad Tecmilenio**
