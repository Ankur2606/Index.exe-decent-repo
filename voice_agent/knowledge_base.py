import os
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Global variables for client, collection and chroma client
client = None
collection = None
chroma_client = None

def get_genai_client():
    global client
    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = genai.Client()
    return client

def get_embedding(text: str, title: str = "none") -> list[float]:
    genai_client = get_genai_client()
    formatted_text = f"title: {title} | text: {text}"
    response = genai_client.models.embed_content(
        model="gemini-embedding-2",
        contents=formatted_text
    )
    return response.embeddings[0].values

def get_query_embedding(query: str) -> list[float]:
    genai_client = get_genai_client()
    formatted_query = f"task: search result | query: {query}"
    response = genai_client.models.embed_content(
        model="gemini-embedding-2",
        contents=formatted_query
    )
    return response.embeddings[0].values

def build_knowledge_base():
    global collection, chroma_client
    
    persist_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "data", "chroma_db")
    os.makedirs(persist_dir, exist_ok=True)
    
    try:
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        collection = chroma_client.get_or_create_collection(
            name="bengaluru_traffic",
            metadata={"hnsw:space": "cosine"}
        )
        if collection.count() > 0:
            print(f"ChromaDB persistent collection loaded successfully with {collection.count()} chunks. Skipping rebuild.")
            return
    except Exception as e:
        print(f"ChromaDB corrupt or unreadable ({e}). Wiping and rebuilding...")
        # Wipe the corrupt database directory
        import shutil
        shutil.rmtree(persist_dir, ignore_errors=True)
        os.makedirs(persist_dir, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        collection = chroma_client.get_or_create_collection(
            name="bengaluru_traffic",
            metadata={"hnsw:space": "cosine"}
        )

    corpus = [
        # Category 1: Major Corridors and Their Properties
        {
            "text": "Outer Ring Road East is 17.1 kilometers long connecting Silk Board to KR Puram. It houses over 500 technology companies and serves 8 lakh daily commuters. A major corridor redesign is underway for the year 2026. Peak hour traffic speed is typically 10 kilometers per hour. Key bottlenecks include Bellandur and Marathahalli. Alternate routes are Wind Tunnel Road or inner roads through HSR Layout.",
            "category": "corridor",
            "name": "Outer Ring Road East"
        },
        {
            "text": "Outer Ring Road West connects Tumkur Road to Mysore Road. It is a vital commercial transit artery experiencing heavy freight vehicle movement. Major bottlenecks occur near Gorguntepalya and Nayandahalli. Peak traffic occurs between 8 am and 11 am and between 5 pm and 9 pm. Commuters should use Magadi Road as a parallel alternate route.",
            "category": "corridor",
            "name": "Outer Ring Road West"
        },
        {
            "text": "Hosur Road is the Electronic City corridor. It features an elevated tollway stretch and a busy lower deck road. Major bottlenecks happen at the Silk Board junction and the Electronic City toll plaza. Alternate routes include the NICE Road or Hosa Road to bypass the primary traffic stream.",
            "category": "corridor",
            "name": "Hosur Road"
        },
        {
            "text": "Tumkur Road also known as National Highway 4 is a critical entrance path to Bengaluru from northern Karnataka. It features heavy industrial cargo movement. Major bottlenecks include Yeshwantpur circle and Peenya industrial zone. Alternate transit routes are via HMT Road or outer bypass roads.",
            "category": "corridor",
            "name": "Tumkur Road"
        },
        {
            "text": "Bellary Road also known as National Highway 44 is the Airport Road corridor. High speed passenger vehicles and airport cabs dominate this corridor. Major bottlenecks are the Hebbal flyover and Mekhri Circle. Alternate routes are via Hennur Road or Thanisandra Main Road.",
            "category": "corridor",
            "name": "Bellary Road"
        },
        {
            "text": "Mysore Road also known as National Highway 275 is a southwest connection route. It is congested near the satellite bus terminal and Nayandahalli. Alternate routes include Magadi Road or NICE Road to bypass the central corridor.",
            "category": "corridor",
            "name": "Mysore Road"
        },
        {
            "text": "Old Madras Road also known as National Highway 75 is an eastern connection route. Bottlenecks occur near Tin Factory junction and KR Puram bridge. Metro construction causes ongoing delays. Alternate routes are via Outer Ring Road or Hennur bypass.",
            "category": "corridor",
            "name": "Old Madras Road"
        },
        {
            "text": "MG Road and Brigade Road are the Central Business District corridors. High pedestrian counts and commercial shopping activity restrict vehicle speeds. Key bottlenecks occur near Anil Kumble circle and Cauvery junction. Alternate routes include Residency Road or Richmond Road.",
            "category": "corridor",
            "name": "MG Road and Brigade Road"
        },
        {
            "text": "Sarjapur Road is a key IT corridor connecting southeastern suburbs to the city center. It has narrow lanes and high residential commute volume. Bottlenecks occur at Kaikondrahalli and Carmelaram railway crossing. Alternate routes are via Haralur Road or Outer Ring Road.",
            "category": "corridor",
            "name": "Sarjapur Road"
        },
        {
            "text": "Bannerghatta Road is a major north south connection. It suffers from metro construction works and narrow stretches. Bottlenecks occur near Jayadeva flyover and Gottigere. Alternate routes are via Kanakapura Road or Hosur Road.",
            "category": "corridor",
            "name": "Bannerghatta Road"
        },

        # Category 2: Known Congestion Hotspots
        {
            "text": "Silk Board Junction is Asia's most congested convergence point. The double decker flyover is still incomplete. Peak delay times exceed 30 minutes. Alternate routes are HSR Layout sector 1 roads or Madiwala inner lanes.",
            "category": "hotspot",
            "name": "Silk Board Junction"
        },
        {
            "text": "KR Puram bridge is a single lane bottleneck caused by heavy bus traffic and metro construction. Delay time is typically 25 minutes. Alternate routes are Outer Ring Road bypass or Hoodi road.",
            "category": "hotspot",
            "name": "KR Puram bridge"
        },
        {
            "text": "Marathahalli junction is a critical intersection on the Outer Ring Road East. It suffers from heavy pedestrian crossings and service lane congestion. Alternate routes are HAL Old Airport Road or outer campus access points.",
            "category": "hotspot",
            "name": "Marathahalli junction"
        },
        {
            "text": "Hebbal flyover is a major bottleneck on Bellary Road caused by lane merging from the outer ring road. Peak delay is 20 minutes. Alternate route is Outer Ring Road to Hennur road.",
            "category": "hotspot",
            "name": "Hebbal flyover"
        },
        {
            "text": "Tin Factory junction is a high density bottleneck on Old Madras Road. It has narrow lanes and heavy intercity bus operations. Delay is 20 minutes. Alternate route is via Kasturi Nagar roads.",
            "category": "hotspot",
            "name": "Tin Factory junction"
        },
        {
            "text": "Yeshwantpur circle connects Tumkur road to city center. It experiences congestion near the railway station and market area. Alternate route is via Peenya main road.",
            "category": "hotspot",
            "name": "Yeshwantpur circle"
        },
        {
            "text": "Mekhri Circle on Bellary Road is congested due to airport traffic and Palace grounds events. Alternate route is via Jayamahal Road.",
            "category": "hotspot",
            "name": "Mekhri Circle"
        },
        {
            "text": "Richmond Circle connects central districts to residential areas. It has multilane signal cycles causing peak hour queues. Alternate route is via Langford Town roads.",
            "category": "hotspot",
            "name": "Richmond Circle"
        },
        {
            "text": "Banashankari junction is a major transit interchange hub with metro station and bus terminal. Alternate route is via Padmanabhanagar main road.",
            "category": "hotspot",
            "name": "Banashankari junction"
        },
        {
            "text": "HSR Layout junction is congested due to retail market traffic and software park commuters. Alternate route is via Haralur road.",
            "category": "hotspot",
            "name": "HSR Layout junction"
        },
        {
            "text": "Agara junction connects HSR Layout to Sarjapur road and Outer Ring Road. It suffers from lane merging bottlenecks. Alternate route is via Iblur junction.",
            "category": "hotspot",
            "name": "Agara junction"
        },
        {
            "text": "Bellandur junction is a major pressure point on Outer Ring Road East. It is surrounded by large tech parks. Peak delays reach 25 minutes. Alternate route is via Panathur road.",
            "category": "hotspot",
            "name": "Bellandur junction"
        },
        {
            "text": "Mahadevapura stretch is a heavy traffic corridor on Outer Ring Road. It has multiple software park entry points. Alternate route is via Hoodi main road.",
            "category": "hotspot",
            "name": "Mahadevapura stretch"
        },
        {
            "text": "Electronic City toll plaza is a major toll collection bottleneck on Hosur Road. Elevated toll queues build up quickly. Alternate route is lower deck Hosur road.",
            "category": "hotspot",
            "name": "Electronic City toll plaza"
        },

        # Category 3: Event-Type to Traffic Impact Mapping
        {
            "text": "Vehicle breakdown events typically cause moderate congestion with an impact radius of 1 kilometer. Average clearing duration is 45 minutes. Deploy 2 officers for manual guidance. Diversion is rarely needed unless heavy commercial trucks block lanes.",
            "category": "event_impact",
            "name": "vehicle_breakdown"
        },
        {
            "text": "Accidents cause high traffic congestion with an impact radius of 2 kilometers. Average duration is 60 to 90 minutes. Deploy 4 officers and towing cranes immediately. Localized lane diversion is required to redirect traffic flow away from the collision zone.",
            "category": "event_impact",
            "name": "accident"
        },
        {
            "text": "Water logging causes severe gridlock with an impact radius of 3 kilometers. Average duration is 3 to 5 hours depending on rain intensity. Deploy 6 officers and pump trucks. Standard diversions are highly required as entire underpasses block traffic flow.",
            "category": "event_impact",
            "name": "water_logging"
        },
        {
            "text": "Construction work causes long term traffic slow downs. Impact radius is 1 kilometer. Average duration is weeks to months. Deploy 3 officers to guide commuters around barriers. Permanent barricades and warning signage must be installed. Alternate route diversion is recommended.",
            "category": "event_impact",
            "name": "construction"
        },
        {
            "text": "Public rallies cause critical congestion with an impact radius of 4 kilometers. Duration can exceed 4 hours. Deploy 10 officers. Major route diversions are mandatory to protect the march corridor and maintain flow.",
            "category": "event_impact",
            "name": "public_rally"
        },
        {
            "text": "VIP movement causes high congestion with an impact radius of 2 kilometers. It lasts for 30 minutes. Deploy 5 officers for rolling block control. Short term rolling road closures are used with no permanent diversions.",
            "category": "event_impact",
            "name": "vip_movement"
        },
        {
            "text": "Tree fallen events cause high congestion with an impact radius of 1.5 kilometers. Average clearing duration is 2 hours. Deploy 3 officers and forest clearing crews. Local lane bypass is required.",
            "category": "event_impact",
            "name": "tree_fallen"
        },
        {
            "text": "Fire incidents cause critical congestion with an impact radius of 3 kilometers. Average duration is 3 hours. Deploy 8 officers and emergency fire trucks. Full road closure and wide perimeter diversions are mandatory.",
            "category": "event_impact",
            "name": "fire_incident"
        },

        # Category 4: Diversion Route Pairs
        {
            "text": "Outer Ring Road East diversion route: For incidents at Bellandur, direct traffic to divert via Sarjapur Road and Haralur Road. This alternate path adds approximately 15 minutes of travel time.",
            "category": "diversion_pair",
            "name": "Outer Ring Road East Bellandur"
        },
        {
            "text": "Outer Ring Road East diversion route: For incidents at Marathahalli, direct traffic to divert via HAL Old Airport Road and Wind Tunnel Road. This alternate path adds approximately 20 minutes of travel time.",
            "category": "diversion_pair",
            "name": "Outer Ring Road East Marathahalli"
        },
        {
            "text": "Outer Ring Road West diversion route: For incidents near Gorguntepalya, divert vehicles via Magadi Road and Pipeline road. This alternate path adds 18 minutes of travel time.",
            "category": "diversion_pair",
            "name": "Outer Ring Road West Gorguntepalya"
        },
        {
            "text": "Hosur Road diversion route: For incidents near Silk Board, divert vehicles via HSR Layout sector 1 and Agara. This adds 25 minutes of travel time.",
            "category": "diversion_pair",
            "name": "Hosur Road Silk Board"
        },
        {
            "text": "Bellary Road diversion route: For incidents near Hebbal flyover, divert vehicles via Hennur Road and Thanisandra Main Road. This adds 15 minutes of travel time.",
            "category": "diversion_pair",
            "name": "Bellary Road Hebbal"
        },

        # Category 5: Police Jurisdiction Map
        {
            "text": "East Zone 1 traffic jurisdiction covers major roads including HAL Old Airport Road, Indiranagar, and parts of Outer Ring Road East. Key police stations are HAL Traffic Police Station and Indiranagar Traffic Police Station.",
            "category": "jurisdiction",
            "name": "East Zone 1"
        },
        {
            "text": "East Zone 2 traffic jurisdiction covers Bellandur, Marathahalli, and Whitefield areas. Key police stations include Bellandur Traffic Police Station and Mahadevapura Traffic Police Station.",
            "category": "jurisdiction",
            "name": "East Zone 2"
        },
        {
            "text": "South Zone traffic jurisdiction covers HSR Layout, Koramangala, and Jayanagar areas. Key stations include HSR Layout Traffic Police Station and Koramangala Traffic Police Station.",
            "category": "jurisdiction",
            "name": "South Zone"
        },

        # Category 6: Peak Hour Patterns
        {
            "text": "Bengaluru daily traffic follows distinct peak hour patterns. Morning peak operates from 7:30 am to 10:30 am when office commuters head to IT parks. Evening peak operates from 5:30 pm to 9:00 pm. Saturdays show afternoon shopping peaks. Public holidays and IT sector holidays reduce traffic volume by 40 percent.",
            "category": "peak_hours",
            "name": "Daily Peak Patterns"
        },

        # Category 7: Namma Metro Impact (2026)
        {
            "text": "Namma Metro construction in 2026 causes major traffic disruptions. The Blue Line construction along Outer Ring Road from KR Puram to Silk Board reduces active road lanes by 50 percent. Commuters must use alternate arterial routes to avoid heavy congestion near metro stations.",
            "category": "metro_impact",
            "name": "Blue Line Construction"
        }
    ]

    print("Embedding traffic corpus into ChromaDB knowledge base...")
    
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, item in enumerate(corpus):
        text = item["text"]
        title = item.get("name", "none")
        vector = get_embedding(text, title=title)
        
        ids.append(f"chunk_{i}")
        embeddings.append(vector)
        documents.append(text)
        metadatas.append({
            "category": item["category"],
            "name": item["name"]
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Successfully loaded {len(corpus)} traffic knowledge chunks into ChromaDB.")

def retrieve_traffic_knowledge(query: str, top_k: int = 3) -> list[str]:
    global collection
    if collection is None:
        # Build if not initialized
        build_knowledge_base()
        
    try:
        query_vector = get_query_embedding(query)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        # Check if results are empty or none
        if not results or "documents" not in results or len(results["documents"][0]) == 0:
            return []
            
        return results["documents"][0]
    except Exception as e:
        print(f"RAG query lookup error: {e}")
        return []

if __name__ == "__main__":
    # Test execution
    build_knowledge_base()
    results = retrieve_traffic_knowledge("Outer Ring Road East Bellandur water logging")
    print("Test Query Results:")
    for doc in results:
        print("*", doc)
