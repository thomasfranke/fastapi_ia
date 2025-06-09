### Criar Ambiente
python3 -m venv .env

### Iniciar o ambiente
source .env/bin/activate
deactivate

### Inicializar
uvicorn main:app --reload --root-path . 

### Endpoints

#### Age Rating Endpoint
`POST`: `/ia/age_classification`
```
{
  "text": "string"
}
```

#### Hate Speech Detection
`POST`: `/ia/hate_speech/detect`
```
{
  "text": "string"
}
```

#### Analyze Hate Speech
`POST`: `/ia/hate_speech/analyze`
```
{
  "text": "string"
}
```

### Folder Structure
```
fastapi_ia/
├── .env/                          # Arquivos de variáveis de ambiente (.env)
│
├── app/
│   ├── data/                      # Implementações de acesso a dados
│   │   ├── models/                # Modelos do ORM (ex: SQLAlchemy)
│   │   └── repositories_impl/     # Implementações das interfaces do domínio
│   │
│   ├── domain/                    # Entidades e regras de negócio
│   │   ├── entities/              # Entidades puras (sem dependências externas)
│   │   ├── repositories/          # Interfaces para repositórios (contratos)
│   │   ├── services/              # Regras de negócio auxiliares (opcional)
│   │   └── usecases/              # Casos de uso (Application Services)
│   ├── infrastructure/            # Huddingface
│   └── presentation/              # Camada de apresentação (FastAPI)
│       ├── feature_1/             # Rotas, controllers e schemas da feature 1
│       │   ├── controller.py
│       │   ├── routes.py
│       │   └── schemas.py
│       └── feature_2/             # Rotas, controllers e schemas da feature 2
│
└── main.py                        # Ponto de entrada da aplicação FastAPI
```