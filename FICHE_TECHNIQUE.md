# Smart AI CCTV Orchestrator — Fiche Technique

## Présentation Produit
Smart AI CCTV Orchestrator est une plateforme de vidéosurveillance intelligente, conçue pour analyser en temps réel des flux vidéo (YouTube, RTSP, CCTV) et fournir des descriptions d’activité claires et exploitables. Le système combine vision par ordinateur (YOLOv11) et raisonnement LLM pour transformer des détections brutes en informations opérationnelles.

## Objectif
Offrir une solution de surveillance proactive, performante et extensible, capable de:
- Détecter des objets et comportements
- Décrire la scène en langage naturel
- Résumer l’activité par intervalles réguliers
- Se connecter facilement à des applications web (React)

## Points Clés
- **Analyse en temps réel** des vidéos et caméras
- **Sortie textuelle claire** pour opérateurs et dashboards
- **Résumés LLM** périodiques pour interprétation
- **Architecture modulaire** pour évoluer rapidement
- **Optimisé CPU/Edge** pour déploiement sur machines légères

## Fonctionnalités Principales
- Ingestion vidéo (YouTube / RTSP)
- Détection d’objets avec YOLOv11 + tracking
- Extraction de vecteurs par frame (class_id, conf, bbox, track_id)
- Génération de descriptions de scène en texte
- Analyse LLM périodique (toutes les 8 frames)
- Diffusion en temps réel via SSE
- Arrêt manuel des jobs en cours

## Flux de Données
1. **Flux vidéo** → YOLOv11
2. **Vecteurs de détection** → Logic Engine
3. **Texte de scène** → LLM (Scaleway Qwen)
4. **Résultats** → UI (SSE streaming)

## Architecture
```
/backend
  /app
    /api
      /v1
        /endpoints
    /services
      /vision_engine
      /llm_engine
      /camera_manager
    /core
  main.py
```

## Endpoints (API)
- `POST /api/v1/vision/detect-youtube`
- `GET /api/v1/vision/jobs/{job_id}/stream`
- `POST /api/v1/vision/jobs/{job_id}/stop`
- `POST /api/v1/llm/chat`

## Sorties (UI)
Le front reçoit:
- **Batchs de 4 frames** (vecteurs)
- **Texte de scène** pour chaque batch
- **Analyse LLM** toutes les 8 frames

## Performances
Conçu pour:
- CPU (Edge devices)
- Streaming optimisé (SSE)
- Traitement par batch pour limiter la charge

## Sécurité & Extensibilité
- Variables d’environnement pour clés API
- Modules séparés (vision, LLM, camera)
- Prêt pour intégration zone fencing, alertes, rules engine

## Cas d’Usage
1. Surveillance d’entrepôts
2. Sécurisation de commerces
3. Monitoring d’espaces publics
4. Analyse d’activités suspectes
5. Pilotage de centres de contrôle

## Positionnement
