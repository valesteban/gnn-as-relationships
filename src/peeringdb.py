import urllib.request
from typing import Dict, Set, Tuple
import requests

PEERINGDB_API = "https://www.peeringdb.com/api"
REQUEST_TIMEOUT = 30


def peeringdb_get_all(resource: str, **params) -> list[dict]:
    """
    Hace GET paginado a la API de PeeringDB.
    """
    all_rows = []
    limit = 250
    skip = 0

    while True:
        query = {"limit": limit, "skip": skip, **params}

        response = requests.get(
            f"{PEERINGDB_API}/{resource}",
            params=query,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        payload = response.json()
        rows = payload.get("data", [])

        if not rows:
            break

        all_rows.extend(rows)

        if len(rows) < limit:
            break

        skip += limit

    return all_rows


def get_ix_ids_by_country(country_code: str = "CL") -> Set[int]:
    """
    Obtiene IDs de IXPs de un país.
    """
    country_code = country_code.upper()
    rows = peeringdb_get_all("ix", country=country_code, fields="id,name,country")
    ix_ids = {int(row["id"]) for row in rows if row.get("id") is not None}

    print(f"[OK] IXPs en {country_code}: {len(ix_ids)}")
    return ix_ids


def get_ixp_participant_asns(country_code: str = "CL") -> Set[str]:
    """
    Obtiene ASNs que participan en IXPs del país.
    Estrategia robusta:
    1) obtener ix IDs del país
    2) consultar netixlan por ix_id
    """
    ix_ids = get_ix_ids_by_country(country_code)
    participant_asns: Set[str] = set()

    for ix_id in ix_ids:
        rows = peeringdb_get_all("netixlan", ix_id=ix_id, fields="asn,ix_id")
        for row in rows:
            asn = row.get("asn")
            if asn is not None:
                participant_asns.add(str(asn))

    print(f"[OK] ASNs participantes en IXPs de {country_code}: {len(participant_asns)}")
    return participant_asns


def get_facility_ids_by_country(country_code: str = "CL") -> Set[int]:
    """
    Obtiene IDs de facilities de un país.
    """
    country_code = country_code.upper()
    rows = peeringdb_get_all("fac", country=country_code, fields="id,name,country")
    fac_ids = {int(row["id"]) for row in rows if row.get("id") is not None}

    print(f"[OK] Facilities en {country_code}: {len(fac_ids)}")
    return fac_ids


def get_facility_asns(country_code: str = "CL") -> Set[str]:
    """
    Obtiene ASNs con presencia en facilities del país.
    Usa netfac -> net_id -> net(asn)
    """
    fac_ids = get_facility_ids_by_country(country_code)
    facility_asns: Set[str] = set()
    net_id_to_asn: Dict[int, str] = {}

    for fac_id in fac_ids:
        rows = peeringdb_get_all("netfac", fac_id=fac_id, fields="net_id,fac_id")

        for row in rows:
            net_id = row.get("net_id")
            if net_id is None:
                continue

            net_id = int(net_id)

            if net_id not in net_id_to_asn:
                net_rows = peeringdb_get_all("net", id=net_id, fields="id,asn")
                if not net_rows:
                    continue

                asn = net_rows[0].get("asn")
                if asn is None:
                    continue

                net_id_to_asn[net_id] = str(asn)

            facility_asns.add(net_id_to_asn[net_id])

    print(f"[OK] ASNs con presencia en facilities de {country_code}: {len(facility_asns)}")
    return facility_asns


def download_peeringdb_file(output_path: str, year: str, month: str):
    """
    Descarga el archivo completo de ASNs de PeeringDB.
    """
    
    url = f"https://publicdata.caida.org/datasets/peeringdb/{year}/{month}/peeringdb_2_dump_{year}_{month}_01.json"
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"[OK] Archivo de PeeringDB descargado: {output_path}")
    except Exception as e:
        print(f"[ERROR] No se pudo descargar el archivo de PeeringDB: {e}")
