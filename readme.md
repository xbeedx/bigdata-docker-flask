# Projet TuringGPT

Ce projet implémente un chatbot basé sur l'API OpenAI, avec une persistance des données via MongoDB et Redis, le tout conteneurisé avec Docker.

## Prérequis

- Docker et Docker Compose doivent être installés sur votre machine.
- Une clé API valide pour l'API OpenAI.

## Mise en place de l'environnement

Pour démarrer le projet, veuillez suivre les étapes suivantes :

1. **Cloner le dépôt :**

```bash
git clone git@github.com:xbeedx/bigdata-docker.git
cd bigdata-docker
```

2. **Configurer les variables d'environnement :**

- Créez un fichier .env à la racine du projet et ajoutez votre clé API OpenAI :

```bash    
OPENAI_API_KEY=votre_clé_api
```

3. **Construire et démarrer les containers :**

```bash
docker-compose up --build -d
```

- Cela va construire les images nécessaires et démarrer les containers pour l'application web, MongoDB et Redis.

4. **Utilisation de l'application**

- Accédez à l'application via http://localhost:5000 sur votre navigateur.
- Interagissez avec le chatbot via l'interface utilisateur web.

5. **Accès aux bases de données**

- Pour accéder aux shells interactifs de MongoDB et Redis :

    - MongoDB :

```bash
docker exec -it bigdata-docker-mongo-1 mongosh
```
- Redis :

```bash
docker exec -it bigdata-docker-redis-1 redis-cli
```

6. **Vérification des données**

- MongoDB :

    - Pour vérifier les conversations stockées dans MongoDB :

```bash
use gpt_chat;
db.conversations.find().pretty();
```

- Redis :

    - Pour récupérer la dernière réponse du bot stockée dans Redis :

```bash
KEYS * 
GET [NOM_DE_LA_CLE]
```

8. **Arrêt de l'environnement**

- Pour arrêter et supprimer les containers, utilisez :

```bash
docker-compose down
```

7. **Nettoyage**

- Pour supprimer les volumes et les données persistance, exécutez :

```bash
docker-compose down --volumes
```
