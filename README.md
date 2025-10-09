# LaLiga RAG SQL – MVP

Prototipo mínimo de RAG (SQL QA) con LangChain para consultar una base de datos local de LaLiga y generar análisis e informes.

## Estructura del proyecto
- `src/` → código Python  
- `data/` → tu base de datos `.sqlite` y descripciones de tablas  
- `reports/` → informes generados  

## Uso rápido (con Codespaces)
1. Crea un Codespace desde **Code → Codespaces → Create on main**  
2. Copia `.env.example` a `.env` y añade tu API key de OpenAI  
3. Sube tu base `laliga.sqlite` a `data/`  
4. Ejecuta:

