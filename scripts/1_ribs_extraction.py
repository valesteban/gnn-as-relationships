from _pybgpstream import BGPStream, BGPRecord
import datetime
import time
import argparse


"""
1.- Este archivo se encarga de extraer los AS PATHS de las RIBs BGP de los colectores de RouteViews y RIPE RIS.
Para esto se utiliza la librería bgpstream, que nso permite descargar la informacion historicas de BGP.
Se puede especificar la fecha de inicio y duración, agregar filtros , seleccionar colectores u projectos especificos, etc.
"""
LOG_EVERY_N = 1000000


def ribs_download(start_date: str,
                  duration:str,
                  data_path:str
                  ):

    base_ts = int(datetime.datetime.strptime(start_date, '%m/%d/%Y').strftime('%s'))
    end_ts = base_ts + int(duration)

    # Formato: YYYYMMDD-HHMM
    start_str = datetime.datetime.fromtimestamp(base_ts).strftime('%Y%m%d_%H%M')
    end_str = datetime.datetime.fromtimestamp(end_ts).strftime('%Y%m%d_%H%M')

    # Create a new bgpstream instance and a reusable bgprecord instance
    stream = BGPStream()
    stream.add_interval_filter(base_ts,end_ts)
    stream.add_filter('record-type', 'ribs')
    stream.start()
    
    print(f'Downloading BGP paths from {start_str} to {end_str}:')
    path_set = set()
    filename = f'rib_{start_str}_to_{end_str}.txt'
    f = open(data_path + filename, 'w')
    count_records = 0
    count_as_paths = 0
    print('Downloading Paths...')

    t0 = time.perf_counter()
    while True:
        count_records +=1
        rec = stream.get_next_record()
        if rec is None:
            print('Paths Downloaded')
            break
        if rec.status != "valid":
            continue
        else:
            elem = rec.get_next_elem()
            while(elem):
                path = elem.fields['as-path']
                if '{' in path or '(' in path:
                    elem = rec.get_next_elem()
                    continue
                prefix = elem.fields['prefix']
                # Focus on IPv4 prefixes
                if ":" not in prefix and path not in path_set:
                    f.write(path.replace(' ', '|') + '\n')
                    count_as_paths +=1
                    path_set.add(path)
                elem = rec.get_next_elem()
        
        if count_records % LOG_EVERY_N == 0:
            f.flush()
            print(f'records: {str(count_records)}\r')
    t1 = time.perf_counter()
    print(f"Tiempo de ejecución: {t1 - t0:.4f} segundos")
    print(f" {count_records} REC total extracted")
    print(f" {count_as_paths} AS PATHS total extracted")
    f.close()


if __name__ == '__main__':

    

    print("#######################")
    print("RIBs EXTRACTION")
    print("#######################")



    # FORMATO: MM/DD/YYYY
    
    start = "03/01/2026" 

    list_durations = [ 1800, # 30 min
                       3600, # 1 hr
                       7200,  # 2 hr
                       14400, # 4 hr
                       21600, # 6 hrs
                       43200, # 12 hrs
                       86400, # 24 hrs
                       ] 
        

    for duration in list_durations:

        print("#####################################################")
        print(f"Duration : {duration}")
        print("#####################################################")
        
        ribs_download(start, duration, data_path = "/media/vale/KINGSTON/TESIS/data_2026/")

