# Python File With Necessary Configurations and Constants

# Label Names and Supplementary Code

import torch
from pathlib import Path
import threading

class ModelConfig:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Date configurations - initialize with defaults
        self._min_year = 1990
        self._max_year = 2025
        self._num_months = 12

        self.BOND_BINS = {
            'not_set_or_missing': -1,
            'set_at_0': 0,
            'set_0_to_1000': 1,
            'set_1000_to_5000': 2,
            'set_5000_to_10000': 3,
            'set_10000_to_25000': 4,
            'set_25000_to_50000': 5,
            'set_50000_to_100000': 6,
            'set_over_100000': 7
        }
        self.NUM_BOND_CLASSES = 9

        self.LABEL_NAMES = [
            "FIRST_APPEARANCE_HEARING_OCCURRED",
            "BOND_SET",
            "BOND_NOT_SET",
            "DEFENDANT_RELEASED_ON_OWN_RECOGNIZANCE",
            "THE_COURT_APPOINTED_THE_ORLEANS_PUBLIC_DEFENDER",
            "ANKLE_MONITORING_IMPOSED",
            "DOMESTIC_STAY_AWAY_ORDER_IMPOSED",
            "CHARGES_ACCEPTED",
            "CHARGES_REFUSED",
            #"GWENS_LAW_HEARING_OCCURRED",
            "PROBABLE_CAUSE_FOUND",
            "PROBABLE_CAUSE_NOT_FOUND",
            "DEFENDANT_IN_CUSTODY_AND_NOT_IN_COURT",
            "THE_STATE_WAS_NOT_PRESENT",
            "PRE_TRIAL_CONFERENCE_HELD",
            "CONTINUANCE_OCCURRED",
            "PLED_GUILTY_TO_CHARGES",
            "PLED_NOT_GUILTY_TO_CHARGES",
            "TRIAL_VERDICT_REACHED",
            "SENTENCE_THAT_INCLUDES_JAIL_TIME_WAS_IMPOSED",
            "SENTENCE_THAT_DOES_NOT_INCLUDE_JAIL_TIME_WAS_IMPOSED"
        ]
        
        

        self.EXCLUDED_LABELS = {'NONE', 'NO_DECISION', 'NO_SENTENCE_WAS_IMPOSED'}
        self.MAX_ROWS_PER_SHEET = 1048576
        self.MAX_SHEETS_PER_FILE = 255
        self.SAFE_ROWS_PER_SHEET = 1000000
        self.SAFE_SHEETS_PER_FILE = 250
        self.MODEL_BATCH_SIZE = 32
        self.NGRAM_BATCH_SIZE = 10000
        self.MAX_MEMORY_THRESHOLD = 1000000  # in MB
        self.initialized = True
    
    @property
    def min_year(self):
        return self._min_year
    
    @property
    def max_year(self):
        return self._max_year
    
    @property
    def num_months(self):
        return self._num_months
    
    @property
    def num_years(self):
        return self._max_year - self._min_year + 1
    
    @property
    def num_labels(self):
        return len(self.LABEL_NAMES)
    
    @property
    def temporal_dim(self):
        return self.num_months * self.num_years
    
    def update_date_range(self, min_year, max_year):
        with self._lock:
            self._min_year = min_year
            self._max_year = max_year
            print(f"Date range updated: {self._min_year} to {self._max_year}")
    
CONFIG = ModelConfig()

def get_model_config():
    return CONFIG 

EARLY_TERMINATION_LABELS = {
    'CHARGES_REFUSED',
    'PROBABLE_CAUSE_NOT_FOUND'
}
CASE_CONCLUSION_LABELS = {
    'SENTENCE_THAT_INCLUDES_JAIL_TIME_WAS_IMPOSED', 
    'SENTENCE_THAT_DOES_NOT_INCLUDE_JAIL_TIME_WAS_IMPOSED', 
    'TRIAL_VERDICT_REACHED'
}

COMPLETION_LABELS = {
    "CHARGES_REFUSED",
    "PROBABLE_CAUSE_NOT_FOUND",
    "TRIAL_VERDICT_REACHED",
    "SENTENCE_THAT_INCLUDES_JAIL_TIME_WAS_IMPOSED",
    "SENTENCE_THAT_DOES_NOT_INCLUDE_JAIL_TIME_WAS_IMPOSED"
}

# SECTION List Definitions Used for In Custody and Not In Custody & Facilities Search

## SUBSECTION Full Names of the Parish Jails Found in Louisiana

parish_jails_full = [r"ACADIA\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?", r"ACADIA\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"ACADIA\s*PARISH\s*CRIMINAL\s*JUSTICE(?:\s*(?:CENTER|CTR\.?))?", r"ACADIA\s*PARISH\s*JAIL",
                     r"RAYNE\s*CITY\s*JAIL", r"ALLEN\s*PARISH\s*JAIL", r"KINDER\s*CITY\s*JAIL",
                     r"ASCENSION\s*PARISH\s*JAIL", r"ASSUMPTION\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?",
                     r"ASSUMPTION\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"AVOYELLES\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?",
                     r"AVOYELLES\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"AVOYELLES\s*MARKSVILLE\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"AVOYELLES\s*MARKSVILLE\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?",
                     r"AVOYELLES\s*WOMEN'S\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"BEAUREGARD\s*PARISH\s*JAIL",
                     r"BIENVILLE\s*PARISH\s*JAIL", r"BOSSIER\s*MAXIMUM\s*SECURITY(?:\s*(?:CENTER|CTR\.?))?",
                     r"BOSSIER\s*MEDIUM\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"BOSSIER\s*PARISH\s*JAIL",
                     r"CADDO\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"CALCASIEU\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"CALCASIEU\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"CALCASIEU\s*SHERIFF'S\s*PRISON",
                     r"SULPHUR\s*CITY\s*JAIL", r"DEQUINCY\s*CITY\s*JAIL", r"CALDWELL\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"CALDWELL\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?", r"CAMERON\s*PARISH\s*JAIL",
                     r"CATAHOULA\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"CATAHOULA\s*PARISH\s*JAIL",
                     r"CLAIBORNE\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?", r"CLAIBORNE\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"CLAIBORNE\s*PARISH\s*JAIL", r"CONCORDIA\s*PARISH\s*CORR(?:ECTIONAL|\.)?\s*(?:\s*FACILITY)?",
                     r"CONCORDIA\s*PARISH\s*JAIL", r"RIVER\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"CONCORDIA\s*PARISH\s*CORR(?:ECTIONAL|\.)?\s*FACILITY\s*FOR\s*WOMEN",
                     r"DESOTO\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?", r"DESOTO\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"EAST\s*BATON\s*ROUGE\s*PARISH\s*PRISON", r"EAST\s*CARROLL\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?",
                     r"RIVERBEND\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?", r"EAST\s*CARROLL\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"RIVERBEND\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"EAST\s*FELICIANA\s*PARISH\s*JAIL",
                     r"BASILE\s*CITY\s*JAIL", r"EVANGELINE\s*PARISH\s*JAIL", r"EVANGELINE\s*PARISH\s*PRISON",
                     r"MAMOU\s*CITY\s*JAIL", r"PINE\s*PRAIRIE\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"SOUTH\s*LOUISIANA\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?", r"SOUTH\s*LOUISIANA\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"VILLE\s*PLATTE\s*CITY\s*JAIL", r"FRANKLIN\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?",
                     r"FRANKLIN\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"GRANT\s*PARISH\s*DETENTION\s*(?:\s*FACILITY)?",
                     r"GRANT\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?", r"GRANT\s*PARISH\s*CORR(?:ECTIONAL|\.)?\s*(?:\s*FACILITY)?",
                     r"GRANT\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"IBERIA\s*PARISH\s*CRIMINAL\s*JUSTICE\s*(?:\s*FACILITY)?",
                     r"IBERIA\s*PARISH\s*JAIL", r"IBERVILLE\s*PARISH\s*JAIL",
                     r"IBERVILLE\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?",
                     r"IBERVILLE\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"JACKSON\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"JEFF(?:ERSON|\.)?\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"JEFF(?:ERSON|\.)?\s*PARISH\s*JAIL",
                     r"JEFFERSON\s*DAVIS\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"JEFFERSON\s*DAVIS\s*PARISH\s*JAIL",
                     r"LAFAYETTE\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"LAFOURCHE\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?",
                     r"LAFOURCHE\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"LAFOURCHE\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*COMPLEX)?",
                     r"LAFOURCHE\s*PARISH\s*JAIL", r"LA\s*SALLE\s*PARISH\s*JAIL",
                     r"LA\s*SALLE\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"LINCOLN\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?",
                     r"LINCOLN\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"LIVINGSTON\s*PARISH\s*PRISON",
                     r"LIVINGSTON\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"LIVINGSTON\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?",
                     r"LA\s*TRANSITIONAL\s*(?:CENTER|CTR\.?)\s*FOR\s*WOMEN",
                     r"MADISON\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"MADISON\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?", r"MADISON\s*PARISH\s*JAIL",
                     r"BAYOU\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"MADISON\s*PARISH\s*SOUTHERN\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"MOREHOUSE\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?", r"MOREHOUSE\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"MOREHOUSE\s*PARISH\s*JAIL(?:\s*ANNEX)?", r"MOREHOUSE\s*PARISH\s*JAIL",
                     r"NATCHITOCHES\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?",
                     r"NATCHITOCHES\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"ORLEANS\s*JUSTICE(?:\s*(?:CENTER|CTR\.?))?",
                     r"TEMPORARY\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?", r"TEMPORARY\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"ORLEANS\s*TEMPORARY\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?", r"ORLEANS\s*TEMPORARY\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"ORLEANS\s*PARISH\s*PRISON", r"OUACHITA\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"RICHWOOD\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"PLAQ(?:UEMINES|\.)?\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?",
                     r"PLAQ(?:UEMINES|\.)?\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"PLAQ(?:UEMINES|\.)?\s*PARISH\s*JAIL",
                     r"POINTE\s*COUPEE\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?", r"POINTE\s*COUPEE\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?",
                     r"POINTE\s*COUPEE\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"POINTE\s*COUPEE\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"RAPIDES\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?",
                     r"RAPIDES\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"RAPIDES\s*PARISH\s*DETENTION\s*(?:CENTER|CTR\.?)\s*I", r"RAPIDES\s*PARISH\s*DETENTION\s*(?:CENTER|CTR\.?)\s*II",
                     r"RAPIDES\s*PARISH\s*DETENTION\s*(?:CENTER|CTR\.?)\s*III",
                     r"RAPIDES\s*PARISH\s*CORR(?:ECTIONAL|\.)?\s*(?:CENTER|CTR\.?)\s*I",
                     r"RAPIDES\s*PARISH\s*CORR(?:ECTIONAL|\.)?\s*(?:CENTER|CTR\.?)\s*II",
                     r"RAPIDES\s*PARISH\s*CORR(?:ECTIONAL|\.)?\s*(?:CENTER|CTR\.?)\s*III",
                     r"RAPIDES\s*PARISH\s*JAIL", r"RED\s*RIVER\s*PARISH\s*JAIL",
                     r"RICHLAND\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?",
                     r"RICHLAND\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"SABINE\s*PARISH\s*JAIL",
                     r"SABINE\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?", r"SABINE\s*PARISH\s*WOMEN'S\s*(?:\s*FACILITY)?",
                     r"SABINE\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"ST\.?\s*BERNARD\s*PARISH\s*PRISON",
                     r"ST\.?\s*BERNARD\s*PARISH\s*JAIL", r"ST\.?\s*CHARLES\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"NELSON\s*COLEMAN\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"ST\.?\s*CHARLES\s*PARISH\s*JAIL",
                     r"ST\.?\s*HELENA\s*PARISH\s*JAIL", r"ST\.?\s*JAMES\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?",
                     r"ST\.?\s*JAMES\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"ST\.?\s*JOHN\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"SHERMAN\s*WALKER\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"ST\.?\s*JOHN\s*PARISH\s*PRISON", r"ST\.?\s*LANDRY\s*PARISH\s*JAIL",
                     r"ST\.?\s*MARTIN\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"ST\.?\s*MARTIN\s*SHERIFF'S\s*OFFICE\s*JUVENILE\s*TRAINING(?:\s*(?:CENTER|CTR\.?))?",
                     r"ST\.?\s*MARTIN\s*SHERIFF'S\s*OFFICE\s*TRAINING(?:\s*(?:CENTER|CTR\.?))?", r"PATTERSON\s*CITY\s*JAIL",
                     r"ST\.?\s*MARY\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"ST\.?\s*MARY\s*PARISH\s*LAW\s*ENFORCEMENT(?:\s*(?:CENTER|CTR\.?))?",
                     r"SLIDELL\s*CITY\s*JAIL", r"ST\.?\s*TAMMANY\s*PARISH\s*JAIL", r"AMITE\s*CITY\s*JAIL", r"TANGIPAHOA\s*PARISH\s*JAIL",
                     r"TENSAS\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?", r"TENSAS\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"TERREBONNE\s*CRIMINAL\s*JUSTICE(?:\s*COMPLEX)?", r"TERREBONNE\s*PARISH(?:\s*ANNEX)?", r"TERREBONNE\s*PARISH\s*JAIL",
                     r"UNION\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?", r"UNION\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"VERMILLION\s*PARISH\s*JAIL",
                     r"VERNON\s*PARISH\s*JAIL", r"BOGALUSA\s*CITY\s*JAIL", r"WASHINGTON\s*PARISH\s*JAIL", r"WEBSTER\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?",
                     r"WEBSTER\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"BAYOU\s*DORCHEAT\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"WEBSTER\s*PARISH\s*JAIL",
                     r"WEST\s*BATON\s*ROUGE\s*PARISH\s*JAIL", r"WEST\s*BATON\s*ROUGE\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?",
                     r"WEST\s*BATON\s*ROUGE\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?", r"WEST\s*CARROLL\s*PARISH\s*JAIL",
                     r"WEST\s*FELICIANA\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?", r"WEST\s*FELICIANA\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"WINNFIELD\s*CITY\s*JAIL", r"WINN\s*PARISH\s*DETENTION(?:\s*(?:CENTER|CTR\.?))?", r"WINN\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?",
                     r"WINN\s*PARISH\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?"]

## SUBSECTION Shortened Names/Abbreviations of the Parish Jails Found in Louisiana

parish_jails_short = [r'ACADIA\s*JAIL', r'A\.?P\.?D\.?C\.?', r'ALLEN\s*JAIL', r'A\.?P\.?J\.?', r'ASCENSION\s*JAIL', r'A\.?P\.?J\.?',
                      r'ASSUMPTION\s*JAIL', r'A\.?P\.?D\.?C\.?', r'AVOYELLES\s*JAIL', r'D\.?C\.?-3', r'BEAUREGARD\s*JAIL', r'B\.?P\.?J\.?',
                      r'BIENVILLE\s*JAIL', r'B\.?P\.?J\.?', r'BOSSIER\s*JAIL', r'B\.?M\.?S\.?C\.?', r'B\.?M\.?C\.?C\.?', r'CADDO\s*JAIL', r'C\.?C\.?C\.?',
                      r'CALCASIEU\s*JAIL', r'C\.?P\.?C\.?C\.?', r'C\.?C\.?C\.?', r'CALDWELL\s*JAIL', r'C\.?D\.?C\.?', r'CAMERON\s*JAIL', r'C\.?P\.?J\.?',
                      r'CATAHOULA\s*JAIL', r'C\.?C\.?C\.?', r'CLAIBORNE\s*JAIL', r'C\.?P\.?D\.?C\.?', r'CONCORDIA\s*JAIL', r'C\.?P\.?C\.?C\.?',
                      r'DESOTO\s*JAIL', r'D\.?P\.?D\.?C\.?', r'E\.?B\.?R\.?\s*PARISH\s*PRISON', r'E\.?B\.?R\.?P\.?P\.?', r'E\.?B\.?R\.?\s*JAIL', r'E\.?B\.?R\.?P\.?P\.?',
                      r'EAST\s*CARROLL\s*JAIL', r'E\.?C\.?D\.?C\.?', r'EAST\s*FELICIANA\s*JAIL', r'E\.?F\.?P\.?J\.?', r'EVANGELINE\s*JAIL', r'E\.?P\.?J\.?',
                      r'FRANKLIN\s*JAIL', r'F\.?P\.?D\.?C\.?', r'GRANT\s*JAIL', r'G\.?P\.?D\.?C\.?', r'IBERIA\s*JAIL', r'I\.?P\.?J\.?',
                      r'IBERVILLE\s*JAIL', r'I\.?P\.?J\.?', r'JACKSON\s*JAIL', r'J\.?P\.?C\.?C\.?', r'JEFF(?:ERSON|\.)?\s*JAIL', r'J\.?P\.?C\.?C\.?',
                      r'J\.?P\.?\s*JAIL', r'J\.?P\.?C\.?C\.?', r'JEFFERSON\s*DAVIS\s*JAIL', r'J\.?D\.?P\.?J\.?', r'LAFAYETTE\s*JAIL', r'L\.?P\.?C\.?C\.?',
                      r'LAFOURCHE\s*JAIL', r'L\.?P\.?D\.?C\.?', r'LA\s*SALLE\s*JAIL', r'L\.?S\.?P\.?J\.?', r'LINCOLN\s*JAIL', r'L\.?P\.?D\.?C\.?',
                      r'LIVINGSTON\s*JAIL', r'L\.?P\.?D\.?C\.?', r'MADISON\s*JAIL', r'M\.?P\.?D\.?C\.?', r'MOREHOUSE\s*JAIL', r'M\.?P\.?D\.?C\.?',
                      r'NATCHITOCHES\s*JAIL', r'N\.?P\.?D\.?C\.?', r'O\.?J\.?C\.?', r'O\.?P\.?P\.?', r'OUACHITA\s*JAIL', r'O\.?C\.?C\.?',
                      r'PLAQ(?:UEMINES|\.)?\s*JAIL', r'P\.?P\.?J\.?', r'P\.?P\.?C\.?C\.?', r'POINTE\s*COUPEE\s*JAIL', r'P\.?C\.?D\.?C\.?', r'RAPIDES\s*JAIL', r'D\.?C\.?-I\.?',
                      r'D\.?C\.?-I\.?I\.?', r'D\.?C\.?-I\.?I\.?I\.?',
                      r'RED\s*RIVER\s*JAIL', r'R\.?R\.?P\.?J\.?', r'RICHLAND\s*JAIL', r'R\.?P\.?D\.?C\.?', r'SABINE\s*JAIL', r'S\.?P\.?J\.?',
                      r'ST\.?\s*BERNARD\s*JAIL', r'S\.?B\.?P\.?J\.?', r'ST\.?\s*CHARLES\s*JAIL', r'S\.?C\.?C\.?C\.?', r'ST\.?\s*HELENA\s*JAIL', r'S\.?H\.?P\.?J\.?',
                      r'ST\.?\s*JAMES\s*JAIL', r'S\.?J\.?P\.?D\.?C\.?', r'ST\.?\s*JOHN\s*JAIL', r'S\.?J\.?P\.?C\.?C\.?', r'ST\.?\s*LANDRY\s*JAIL', r'S\.?L\.?P\.?J\.?',
                      r'ST\.?\s*MARTIN\s*JAIL', r'S\.?M\.?P\.?C\.?C\.?', r'ST\.?\s*MARY\s*JAIL', r'S\.?M\.?P\.?C\.?C\.?', r'ST\.?\s*TAMMANY\s*JAIL', r'S\.?T\.?P\.?J\.?',
                      r'TANGIPAHOA\s*JAIL', r'T\.?P\.?J\.?', r'TENSAS\s*JAIL', r'T\.?P\.?D\.?C\.?', r'TERREBONNE\s*JAIL', r'T\.?C\.?J\.?C\.?',
                      r'UNION\s*JAIL', r'U\.?P\.?D\.?C\.?', r'VERMILION\s*JAIL', r'V\.?P\.?J\.?', r'VERNON\s*JAIL', r'V\.?P\.?J\.?',
                      r'WASHINGTON\s*JAIL', r'W\.?P\.?J\.?', r'WEBSTER\s*JAIL', r'W\.?P\.?D\.?C\.?', r'WEST\s*BATON\s*ROUGE\s*JAIL', r'W\.?B\.?R\.?P\.?J\.?',
                      r'WEST\s*CARROLL\s*JAIL', r'W\.?C\.?P\.?J\.?', r'WEST\s*FELICIANA\s*JAIL', r'W\.?F\.?P\.?D\.?C\.?',
                      r'WINN\s*JAIL', r'W\.?P\.?D\.?C\.?', r'W\.?C\.?C\.?', r'WINN',
                      r"RAYNE", r"KINDER", r"MARKSVILLE", r"SULPHUR", r"DEQUINCY", r"RIVER", r"RIVERBEND", r"BASILE", r"MAMOU", r"PINE\s*PRAIRIE",
                      r"SOUTH\s*LOUISIANA", r"SOUTH\s*LOUISIANA", r"VILLE\s*PLATTE", r"BAYOU", r"RICHWOOD", r"NELSON\s*COLEMAN", r"COLEMAN", r"SHERMAN\s*WALKER",
                      r"WALKER", r"PATTERSON", r"SLIDELL", r"AMITE", r"BOGALUSA", r"BAYOU\s*DORCHEAT", r"DORCHEAT", r"WINNFIELD"]

## SUBSECTION Full Names of the State Prisons Found in Louisiana

state_prisons_full = [r'ALLEN\s*CORR(?:ECTIONAL|\.)?\s*(?:CENTER|CTR\.?)?', r'AVOYELLES\s*CORR(?:ECTIONAL|\.)?\s*(?:CENTER|CTR\.?)?', r'RAYMOND\s*LABORDE\s*CORR(?:ECTIONAL|\.)?\s*(?:CENTER|CTR\.?)?',
                      r'(?:B\.?\s*B\.?\s*)?RAYBURN\s*CORR(?:ECTIONAL|\.)?\s*(?:CENTER|CTR\.?)?', r'DAVID\s*WADE\s*CORR(?:ECTIONAL|\.)?\s*(?:CENTER|CTR\.?)?',
                      r'DIXON\s*CORR(?:ECTIONAL|\.)?(?:\s*INSTITUTE|\s*CENTER|\s*CTR\.?)?', r'(?:ELAYN\s*)?HUNT\s*CORR(?:ECTIONAL|\.)?(?:\s*(?:CENTER|CTR\.?))?',
                      r'LOUISIANA\s*CORR(?:ECTIONAL|\.)?\s*INSTITUTE\s*FOR\s*WOMEN', r'LOUISIANA\s*STATE\s*PEN(?:ITENTIARY|\.)?']

## SUBSECTION Shortened Names/Abbreviations of the the State Prisons Found in Louisiana

state_prisons_short = [r'ALLEN', r'A\.?C\.?C\.?', r'AVOYELLES', r'R\.?L\.?C\.?C\.?', r'LABORDE', r'RAYBURN', r'B\.?B\.?R\.?C\.?C\.?', r'DAVID\s*WADE', r'WADE',
                       r'D\.?W\.?C\.?C\.?', r'DIXON',
                       r'D\.?C\.?I\.?', r'D\.?C\.?C\.?', r'HUNT', r'E\.?H\.?C\.?C\.?', r'L\.?C\.?I\.?W\.?', r'WOMEN\'?S\s*PRISON', r'ANGOLA', r'L\.?S\.?P\.?',
                       r'LA\.?\s*CORR(?:ECTIONAL|\.)?\s*INSTITUTE(?:\s*FOR\s*WOMEN)?', r'LA\.?\s*STATE\s*PEN(?:ITENTIARY|\.)?(?:\s*ANGOLA)?' ]

## SUBSECTION Full Names of the Federal Prisons Found in Louisiana

federal_prisons_full = [r'FEDERAL\s*CORR(?:ECTIONAL|\.)?\s*INSTITUTION\s*OAKDALE', r'UNITED\s*STATES\s*PENITENTIARY\s*POLLOCK']

## SUBSECTION Shortened Names/Abbreviations of the the Federal Prisons Found in Louisiana

federal_prisons_short = [r'F\.?C\.?I\.?\s*OAKDALE', r'OAKDALE', r'U\.?S\.?P\.?\s*POLLOCK', r'POLLOCK', r'F\.?C\.?I\.?', r'U\.?S\.?P\.?']

## SUBSECTION List of the Louisiana Parishes
louisiana_parishes = [r'ACADIA', r'ALLEN', r'ASCENSION', r'ASSUMPTION', r'AVOYELLES', r'BEAUREGARD',
                     r'BIENVILLE', r'BOSSIER', r'CADDO', r'CALCASIEU', r'CALDWELL', r'CAMERON', r'CATAHOULA',
                     r'CLAIBORNE', r'CONCORDIA', r'DE\s*SOTO', r'EAST\s*BATON\s*ROUGE', r'EAST\s*CARROLL',
                     r'EAST\s*FELICIANA', r'EVANGELINE', r'FRANKLIN', r'GRANT', r'IBERIA', r'IBERVILLE',
                     r'JACKSON', r'JEFFERSON', r'JEFFERSON\s*DAVIS', r'LA\s*SALLE', r'LAFAYETTE', r'LAFOURCHE',
                     r'LINCOLN', r'LIVINGSTON', r'MADISON', r'MOREHOUSE', r'NATCHITOCHES', r'ORLEANS',
                     r'OUACHITA', r'PLAQUEMINES', r'POINTE\s*COUPEE', r'RAPIDES', r'RED\s*RIVER', r'RICHLAND',
                     r'SABINE', r'ST\.?\s*BERNARD', r'ST\.?\s*CHARLES', r'ST\.?\s*HELENA', r'ST\.?\s*JAMES',
                     r'ST\.?\s*JOHN', r'ST\.?\s*LANDRY', r'ST\.?\s*MARTIN', r'ST\.?\s*MARY', r'ST\.?\s*TAMMANY',
                     r'TANGIPAHOA', r'TENSAS', r'TERREBONNE', r'UNION', r'VERMILION', r'VERNON',
                     r'WASHINGTON', r'WEBSTER', r'WEST\s*BATON\s*ROUGE', r'WEST\s*CARROLL', r'WEST\s*FELICIANA',
                     r'WINN']

## SUBSECTION List of the US States
us_states = [r'ALABAMA', r'ALASKA', r'ARIZONA', r'ARKANSAS', r'CALIFORNIA', r'COLORADO', r'CONNECTICUT',
          r'DELAWARE', r'FLORIDA', r'GEORGIA', r'HAWAII', r'IDAHO', r'ILLINOIS', r'INDIANA', r'IOWA', r'KANSAS',
          r'KENTUCKY', r'LOUISIANA', r'MAINE', r'MARYLAND', r'MASSACHUSETTS', r'MICHIGAN', r'MINNESOTA',
          r'MISSISSIPPI', r'MISSOURI', r'MONTANA', r'NEBRASKA', r'NEVADA', r'NEW\s*HAMPSHIRE', r'NEW\s*JERSEY',
          r'NEW\s*MEXICO', r'NEW\s*YORK', r'NORTH\s*CAROLINA', r'NORTH\s*DAKOTA', r'OHIO', r'OKLAHOMA', r'OREGON',
          r'PENNSYLVANIA', r'RHODE\s*ISLAND', r'SOUTH\s*CAROLINA', r'SOUTH\s*DAKOTA', r'TENNESSEE', r'TEXAS',
          r'UTAH', r'VERMONT', r'VIRGINIA', r'WASHINGTON', r'WEST\s*VIRGINIA', r'WISCONSIN', r'WYOMING']

## SUBSECTION Combining these lists
all_louisiana_jails_full = parish_jails_full + state_prisons_full + federal_prisons_full
all_louisiana_jails_short = parish_jails_short + state_prisons_short + federal_prisons_short
all_louisiana_jails_combined = all_louisiana_jails_full + all_louisiana_jails_short
either_parishes_or_states = louisiana_parishes + us_states

## SUBSECTION Additional o
additional_patterns_facilities = [
    r"(?:(?:ANY\s*)?LOUISIANA\s*|LA\.?\s*|THE\s*)?D\.?O\.?C\.?(?:\s*AT\s*HARD\s*LABOR)?(?:\s*FACILITY)?",
    r"(?:THE\s*)?DEP(?:ARTMENT|T\.?|\.?)?\s*(?:OF\s*)?CORR(?:ECTI(?:O|A)?NS|\.?)?(?:\s*AT\s*HARD\s*LABOR)?(?:\s*FACILITY)?",
    r"(?:O\.?P\.?P\.?|O\.?J\.?C\.?|ORLEANS\s*PARISH\s*PRISON|ORLEANS\s*JUSTICE(?:\s*CENTER|\s*CTR\.?)?)(?:/(?:O\.?P\.?P\.?|O\.?J\.?C\.?|ORLEANS\s*PARISH\s*PRISON|ORLEANS\s*JUSTICE(?:\s*CENTER|\s*CTR\.?)?)|\s*\((?:O\.?P\.?P\.?|O\.?J\.?C\.?|ORLEANS\s*PARISH\s*PRISON|ORLEANS\s*JUSTICE(?:\s*CENTER|\s*CTR\.?)?)\))",    
    r"DIXON\s*C\.?C\.?",
    r"ORLEANS\s*PARIHS\s*PRISON",
    r"ORLEANS\s*PARISH\s*PRISO"
]

total_facilities_for_sentencing = all_louisiana_jails_combined + additional_patterns_facilities






