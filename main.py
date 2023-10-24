import streamlit as st
import numpy as np
from joblib import load
from datetime import datetime, time
import pickle

# 모델 및 레이블 클래스 로드
model = load('./XGBoost_10-23_top6_model.joblib')
label_classes = load('./label_classes.pickle')

# `airport_mapping`정의
# 키: 공항의 코드
# 값: 해당 공항의 전체 이름 
airport_mapping = {'MGM': 'Montgomery Regional Airport',
 'BHM': 'Birmingham-Shuttlesworth International Airport',
 'HSV': 'Huntsville International Airport',
 'MOB': 'Mongomary Regional Airport',
 'ANC': 'Ted Stevens Anchorage International Airport',
 'PHX': 'Phoenix Sky Harbor International Airport',
 'TUS': 'Tucson International Airport',
 'LIT': 'Bill and Hillary Clinton National Airport',
 'XNA': 'Northwest Arkansas National Airport',
 'LAX': 'Los Angeles International Airport',
 'SFO': 'San Francisco International Airport',
 'OAK': 'Oakland International Airport',
 'SAN': 'San Diego International Airport',
 'ORD': 'O Hare International Airport',
 'MDW': 'Chicago Midway International Airport',
 'ATL': 'Hartsfield-Jackson Atlanta International Airport',
 'OGG': 'Kahului Airport',
 'HNL': 'Daniel K. Inouye International Airport',
 'EWR': 'Newark Liberty International Airport',
 'JFK': 'John F. Kennedy International Airport',
 'LGA': 'LaGuardia Airport',
 'SEA': 'Seattle-Tacoma International Airport',
 'SLC': 'Salt Lake City International Airport',
 'DEN': 'Denver International Airport',
 'DFW': 'Dallas/Fort Worth International Airport',
 'IAH': 'George Bush Intercontinental Airport',
 'MIA': 'Miami International Airport',
 'BOS': 'Logan International Airport',
 'LAS': 'McCarran International Airport',
 'MCI': 'Kansas City International Airport',
 'MSP': 'Minneapolis-Saint Paul International Airport',
 'BNA': 'Nashville International Airport',
 'STL': 'St. Louis Lambert International Airport',
 'IAD': 'Washington Dulles International Airport',
 'DCA': 'Ronald Reagan Washington National Airport',
 'BWI': 'Baltimore/Washington International Thurgood Marshall Airport',
 'CLE': 'Cleveland Hopkins International Airport',
 'CVG': 'Cincinnati/Northern Kentucky International Airport',
 'TPA': 'Tampa International Airport',
 'IND': 'Indianapolis International Airport',
 'RDU': 'Raleigh-Durham International Airport',
 'CLT': 'Charlotte Douglas International Airport',
 'PDX': 'Portland International Airport',
 'GEG': 'Spokane International Airport',
 'SMF': 'Sacramento International Airport',
 'MEM': 'Memphis International Airport',
 'SJC': 'Norman Y. Mineta San Jose International Airport',
 'DAL': 'Dallas Love Field',
 'HOU': 'William P. Hobby Airport',
 'FLL': 'Fort Lauderdale-Hollywood International Airport',
 'MCO': 'Orlando International Airport',
 'PIT': 'Pittsburgh International Airport',
 'PHL': 'Philadelphia International Airport',
 'SJU': 'Luis Muñoz Marín International Airport',
 'DHN': 'DHN',
 'BFM': 'BFM',
 'RKS': 'RKS',
 'CPR': 'CPR',
 'COD': 'COD',
 'JAC': 'JAC',
 'GCC': 'GCC',
 'LAR': 'LAR',
 'CYS': 'CYS'}


# 레이블 인코딩 함수
def encode_label(label, label_class):
    return label_class.index(label)


# 공항 정보로 주(State) 반환
def get_state_from_airport(airport):
    state_airports = {
    'Alabama': ['MGM', 'BHM', 'HSV', 'MOB', 'DHN', 'BFM'],
    'Alaska': ['ANC', 'ADQ', 'FAI', 'SCC', 'YAK', 'KTN', 'WRG', 'JNU', 'CDV', 'OME', 'BRW', 'BET', 'DUT', 'OTZ', 'SIT', 'PSG', 'DLG', 'AKN', 'GST', 'ADK'],
    'Arizona': ['PHX', 'TUS', 'YUM', 'AZA', 'FLG', 'PRC'],
    'Arkansas': ['LIT', 'XNA', 'FSM', 'TXK'],
    'California': ['LAX', 'SFO', 'OAK', 'SAN', 'SJC', 'SMF', 'BUR', 'LGB', 'FAT', 'SNA', 'SBP', 'PSP', 'ONT', 'STS', 'MRY', 'ACV', 'SBA', 'SCK', 'RDD', 'BFL', 'MMH', 'SMX'],
    'Colorado': ['COS', 'DEN', 'DRO', 'GJT', 'PUB', 'GUC', 'MTJ', 'EGE', 'HDN', 'ASE'],
    'Connecticut': ['BDL', 'HVN'],
    'Florida': ['TPA', 'DAB', 'FLL', 'MCO', 'SFB', 'RSW', 'MIA', 'GNV', 'PBI', 'VPS', 'ECP', 'MLB', 'JAX', 'PNS', 'SRQ', 'PGD', 'TLH', 'PIE', 'EYW'],
    'Georgia': ['ATL', 'BQK', 'AGS', 'SAV', 'VLD', 'ABY', 'CSG'],
    'Hawaii': ['OGG', 'HNL', 'LIH', 'KOA', 'ITO', 'MKK', 'JHM', 'LNY'],
    'Idaho': ['IDA', 'TWF', 'BOI', 'PIH', 'SUN', 'LWS'],
    'Illinois': ['ORD', 'MDW', 'PIA', 'BMI', 'CMI', 'UIN', 'BLV', 'SPI', 'MLI', 'RFD'],
    'Indiana': ['IND', 'FWA', 'EVV', 'SBN'],
    'Iowa': ['CID', 'DSM', 'DBQ', 'SUX', 'ALO'],
    'Kansas': ['GCK', 'ICT', 'HYS', 'LBL', 'MHK', 'SLN'],
    'Kentucky': ['LEX', 'CVG', 'SDF', 'PAH', 'OWB'],
    'Louisiana': ['MSY', 'BTR', 'LFT', 'SHV', 'MLU', 'AEX', 'LCH'],
    'Maine': ['BGR', 'PWM', 'PQI'],
    'Maryland': ['BWI', 'HGR', 'SBY'],
    'Massachusetts': ['BOS', 'ORH', 'ACK', 'MVY', 'HYA'],
    'Michigan': ['MBS', 'DTW', 'GRR', 'AZO', 'CMX', 'TVC', 'LAN', 'MQT', 'FNT', 'IMT', 'PLN', 'APN', 'CIU', 'MKG', 'ESC'],
    'Minnesota': ['RST', 'MSP', 'DLH', 'BRD', 'BJI', 'HIB', 'INL', 'STC'],
    'Mississippi': ['GTR', 'PIB', 'JAN', 'GPT', 'MEI'],
    'Missouri': ['MCI', 'STL', 'SGF', 'CGI', 'JLN', 'COU', 'BKG'],
    'Montana': ['BIL', 'MSO', 'FCA', 'BZN', 'GTF', 'BTM', 'HLN', 'WYS'],
    'Nebraska': ['OMA', 'LNK', 'GRI', 'BFF', 'LBF', 'EAR'],
    'Nevada': ['LAS', 'RNO', 'EKO'],
    'New Hampshire': ['MHT', 'PSM'],
    'New Jersey': ['EWR', 'ACY', 'TTN'],
    'New Mexico': ['ABQ', 'SAF', 'HOB', 'ROW'],
    'New York': ['BUF', 'JFK', 'LGA', 'ISP', 'ROC', 'SYR', 'ALB', 'PBG', 'HPN', 'ELM', 'ITH', 'SWF', 'BGM', 'ART', 'OGS', 'IAG'],
    'North Carolina': ['CLT', 'RDU', 'OAJ', 'AVL', 'GSO', 'ILM', 'PGV', 'FAY', 'USA', 'EWN'],
    'North Dakota': ['BIS', 'DVL', 'ISN', 'FAR', 'MOT', 'JMS', 'DIK', 'GFK', 'XWA'],
    'Ohio': ['CLE', 'CMH', 'CAK', 'DAY', 'TOL', 'LCK'],
    'Oklahoma': ['OKC', 'LAW', 'TUL', 'SWO'],
    'Oregon': ['PDX', 'MFR', 'RDM', 'OTH', 'EUG'],
    'Pennsylvania': ['PHL', 'PIT', 'SCE', 'ABE', 'ERI', 'MDT', 'AVP', 'IPT', 'LBE'],
    'Puerto Rico': ['SJU', 'BQN', 'PSE'],
    'Rhode Island': ['PVD'],
    'South Carolina': ['CAE', 'CHS', 'GSP', 'HHH', 'MYR', 'FLO'],
    'South Dakota': ['RAP', 'FSD', 'ABR', 'PIR', 'ATY'],
    'Tennessee': ['MEM', 'BNA', 'TYS', 'CHA', 'TRI'],
    'Texas': ['MAF', 'IAH', 'DAL', 'HOU', 'HRL', 'MFE', 'DFW', 'AUS', 'TYR', 'AMA', 'SAT', 'ELP', 'CRP', 'LRD', 'BPT', 'ACT', 'SPS', 'GGG', 'GRK', 'LBB', 'DRT', 'BRO', 'SJT', 'CLL', 'ABI'],
    'U.S. Pacific Trust Territories and Possessions': ['SPN', 'PPG', 'GUM', 'ROP'],
    'U.S. Virgin Islands': ['STT', 'STX'],
    'Utah': ['SLC', 'SGU', 'CDC', 'CNY', 'VEL', 'PVU', 'OGD'],
    'Vermont': ['BTV'],
    'Virginia': ['DCA', 'RIC', 'IAD', 'ORF', 'ROA', 'PHF', 'CHO', 'LYH', 'SHD'],
    'Washington': ['SEA', 'GEG', 'PAE', 'PSC', 'YKM', 'EAT', 'PUW', 'BLI', 'ALW'],
    'West Virginia': ['CRW', 'HTS', 'CKB', 'LWB'],
    'Wisconsin': ['MKE', 'GRB', 'CWA', 'EAU', 'RHI', 'MSN', 'ATW', 'LSE'],
    'Wyoming': ['RKS', 'CPR', 'COD', 'JAC', 'GCC', 'LAR', 'CYS']
    }

    for state, airports in state_airports.items():
        if airport in airports:
            return state
    return None


# 주(State)에서 지역(Region) 변환
def get_region_from_state(state):
    # 딕셔너리 생성: 기후대 별로 9개 + 번외 1개 = 10개
    regions = {
    'Northeast': ["Connecticut", "Delaware", "Maine", "Maryland", "Massachusetts", "New Hampshire", 
                  "New Jersey", "New York", "Pennsylvania", "Rhode Island", "Vermont"],
    'Upper_Midwest': ["Iowa", "Michigan", "Minnesota", "Wisconsin"],
    'Ohio_Valley': ["Illinois", "Indiana", "Kentucky", "Missouri", "Ohio", "Tennessee", "West Virginia"],
    'Southeast': ["Alabama", "Florida", "Georgia", "North Carolina", "South Carolina", "Virginia"],
    'NRP': ["Montana", "Nebraska", "North Dakota", "South Dakota", "Wyoming"],
    'South': ["Arkansas", "Kansas", "Louisiana", "Mississippi", "Oklahoma", "Texas"],
    'Southwest': ["Arizona", "Colorado", "New Mexico", "Utah"],
    'Northwest': ["Idaho", "Oregon", "Washington", "Alaska"],
    'West': ["California", "Nevada", "Hawaii"],
    'etc': ["Puerto Rico", "U.S. Virgin Islands", "U.S. Pacific Trust Territories and Possessions"]
    }

    for region, states in regions.items():
        if state in states:
            return region
    return None

# 날짜를 계절로 변환
def convert_date_to_season(date):
    month = date.month
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'

# 날짜를 월, 일 단위로 반환
def get_month_day(date):
    return date.month, date.day

# 날짜를 1~365 사이의 숫자로 변환('concat_date')
def convert_to_concat_date(date):
    # Days accumulated at the end of each month
    cumulative_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    
    month, day = get_month_day(date)
    return cumulative_days[month - 1] + day

# 시간을 분(0 to 1439) 단위로 변환
def convert_time_to_minutes(time):
    return time.hour * 60 + time.minute

# 시간을 하루 중 어느 시간대인지로 변환
def convert_time_to_part_of_day(minutes):
    if 0 <= minutes < 360:  # 0 to 5:59
        return 'Early Morning'
    elif 360 <= minutes < 720:  # 6:00 to 11:59
        return 'Morning'
    elif 720 <= minutes < 1080:  # 12:00 to 17:59
        return 'Afternoon'
    else:  # 18:00 to 23:59
        return 'Evening'

# 출발/도착 공항 정보로 '거리'(평균) 반환
def get_distance_from_airport(origin_airport, destination_airport):
    DEFAULT_DISTANCE = 784.0979072608434
    with open('./distance_dict.pkl', 'rb') as f:  # 'rb' denotes read in binary mode
        distance_dict = pickle.load(f)
    return distance_dict.get((origin_airport, destination_airport), DEFAULT_DISTANCE)

# 출발/도착 공항 정보로 '비행시간/Time' (평균) 반환
def get_flight_time(origin_airport, destination_airport):
    DEFAULT_FLIGHT_TIME = 140.65164602121513
    with open('./time_dict.pkl', 'rb') as f:  # 'rb' denotes read in binary mode
        time_dict = pickle.load(f)
    return time_dict.get((origin_airport, destination_airport), DEFAULT_FLIGHT_TIME)

# 출발시간, 비행시간 정보로 '도착시간' 반환
def get_arrival_time(origin_airport, destination_airport, departure_time):
    
    # 분으로 변환
    # departure_time_mins = convert_time_to_minutes(departure_time)

    # 비행시간 
    flight_time = get_flight_time(origin_airport, destination_airport)
    
    # 도착시간 계산
    arrival_time = departure_time + flight_time
    if arrival_time >= 1440:  # Simplified the condition to handle == 1440 case as well
        arrival_time -= 1440
    
    return arrival_time

# 항공사의 TailNumber 추출(최빈값)
def get_tail_number(airline):
    tail_number_dict = {
        19393.0: 'N291WN',
        19790.0: 'N983AT',
        20304.0: 'N956SW',
        19977.0: 'N837UA',
        19805.0: 'N955UW',
        20409.0: 'N346JB'
    }
    return tail_number_dict.get(airline, None)

airline_code = {'Southwest Airlines Co.': 19393.0, 
 'SkyWest Airlines Inc.': 20304.0, 
 'American Airlines Inc.': 19805.0,
 'Delta Air Lines Inc.': 19790.0,
 'JetBlue Airways': 20409.0,
 'United Air Lines Inc.': 19977.0,
}

def prepare_input_data(departure_airport, arrival_airport, departure_date, departure_time, airline=airline_code):
    
    departure_time_mins = convert_time_to_minutes(departure_time)

    month, day = get_month_day(departure_date)
    arrival_time_mins = get_arrival_time(departure_airport, arrival_airport, departure_time_mins)
    departure_state = get_state_from_airport(departure_airport)
    arrival_state = get_state_from_airport(arrival_airport)
    distance = get_distance_from_airport(departure_airport, arrival_airport)
    carrier_id = airline['JetBlue Airways']   # 테스트 용
    tail_number = get_tail_number(carrier_id)
    time = get_flight_time(departure_airport, arrival_airport)
    concat_date = convert_to_concat_date(departure_date)
    departure_region = get_region_from_state(departure_state)
    arrival_region = get_region_from_state(arrival_state)
    season = convert_date_to_season(departure_date)
    edt_pod = convert_time_to_part_of_day(departure_time_mins)
    eat_pod = convert_time_to_part_of_day(arrival_time_mins)

    # 배열로 변환
    input_data = np.array([month, day, departure_time_mins, arrival_time_mins, departure_airport, departure_state, 
                           arrival_airport, arrival_state, distance, carrier_id, tail_number, time, 
                           concat_date, departure_region, arrival_region, season, edt_pod, eat_pod])
    
    # 범주형 변수 encoding
    col_idx = [4, 5, 6, 7, 10, 13, 14, 15, 16, 17]
    for idx in col_idx:
        column = list(label_classes.keys())[col_idx.index(idx)]
        input_data[idx] = encode_label(input_data[idx], label_classes[column])

    return input_data

def main():
    abbrev_mapping = {v: k for k, v in airport_mapping.items()}
    st.set_page_config(layout="wide")
    
    st.title("✈️비행기 지연 확률 예측✈️")

    airport_options = ['공항을 선택해주세요.'] + list(airport_mapping.values())

    # 사용자 입력 수집
    departure_airport = st.selectbox("출발 공항", label_classes['Origin_Airport'])
    arrival_airport = st.selectbox("도착 공항", label_classes['Destination_Airport'])
    departure_date = st.date_input("출발 날짜")
    departure_time = st.time_input("출발 시간")
    # Create a list of time objects for every hour
    # hourly_times = [datetime.time(hour=i) for i in range(24)]
    # departure_time = st.selectbox("출발 시간", options=hourly_times)

    # 예측 수행
    import sys
    import io
    import contextlib

    @contextlib.contextmanager
    def suppress_stdout():
        """A context manager to suppress stdout."""
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            yield
        finally:
            sys.stdout = original_stdout

    if st.button("예측하기"):
        input_data = prepare_input_data(departure_airport, arrival_airport, departure_date, departure_time)
        input_data = input_data.astype(np.float32)
        # # 입력 데이터 확장
        # expanded_input_data = expand_input_data(input_data)
        # for item in input_data:
        #     st.write(type(item))


        # 확장된 입력 데이터를 사용하여 예측 수행
        with suppress_stdout():
            probability = model.predict_proba(input_data.reshape(1, -1))

        delay_probability = probability[0][1]

        # 결과 표시
        st.write(f"비행기 지연 확률: {delay_probability:.2%}")

        if delay_probability >= 0.6:
            st.write("높은 확률로 비행이 지연될 수 있습니다.")
        else:
            st.write("높은 확률로 비행이 정시에 출발할 수 있습니다.")

if __name__ == '__main__':
    main()

