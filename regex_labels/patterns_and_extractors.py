import re
import regex
import pandas as pd

from config import all_louisiana_jails_combined, louisiana_parishes, total_facilities_for_sentencing
import string



# SECTION First Appearance Hearing

## Pattern
first_appearance_compiled = re.compile(r'^\s*FIRST\s*APPEARANCE\s*HEARING\s*IN\s*MAGISTRATE\s*COURT\s*SECTION', re.MULTILINE)

## Extractor
def extract_first_appearance_hearing(text):
    if text is None or not isinstance(text, str) or text.strip() == "":
        return "NONE"
    try:
        match = first_appearance_compiled.search(text)
        if match:
            return "FIRST_APPEARANCE_HEARING_OCCURRED"
        else:
            return "NONE"
    except Exception as e:
        return "NONE"


# SECTION Bond Status

## Patterns
### Bond Set
BOND_SET_REGEX = regex.compile(r"""
(?:
    (                              # Group 1: Bond action phrases
        BOND\s+SET                 # "BOND SET"
        |SET\s+BOND                # "SET BOND"  
        |BOND\s+REDUCED            # "BOND REDUCED"
        |REDUCED\s+BOND            # "REDUCED BOND"
        |INCREASED\s+BOND          # "INCREASED BOND"
        |BOND\s+INCREASED          # "BOND INCREASED"
    )
    (?:                            # Non-capturing group for optional prepositions
        \s+                        # Whitespace
        (?:FOR|AT|TO|\@)           # Preposition options
    )?                             # Make preposition group optional
    \s+                            # Required whitespace
    (?:\$)?                        # Optional dollar sign
    ([\d,]+(?:\.\d+)?)             # Group 2: Amount (digits, commas, optional decimal)
    |
    CLERK\'?S\s*OFFICE\s*FILE(?:D)?\s*BAIL\s*ORDER
    |
    (THE\s*)?COURT\s*(?:HAS\s*)?SET\s*BOND(?:\s*ACCORDINGLY|\s*IN\s*THIS\s*MATTER)?
    |
    (THE\s*)?COURT\s*(?:HAS\s*)?(?:ISSUED|COMPLETED|ENTERED)\s*(?:A|AN|THE)?\s*BAIL\s*ORDER
    |
    BAIL\s*ORDER\s*(?:HAS\s*)?(?:BEEN\s*)?(?:ISSUED|COMPLETED|ENTERED)(?:\s*IN\s*THIS\s*MATTER)?
    |
    # JUDGE with preset bond
    (?:JUDGE|COMMISSIONER)\s*(?:[\w-]+(?:\.|\,|\')?\s+){1,1}\s*PRE(?:-)?SET\s*BOND
)
""", regex.VERBOSE | regex.IGNORECASE | regex.MULTILINE)

### Bond Not Set
NO_BOND_SET_REGEX = regex.compile(r"""
(?i)(
    WITHOUT\s+(?:BOND|BAIL)
    |NO\s+(?:BOND|BAIL)
    |NOT\s+ELIGIBLE\s+FOR\s+(?:A\s+)?BOND
    |BOND\s*SET\s*\(\s*NONE\s*\)
)
""", regex.VERBOSE)

## Extractor
def determine_bond_status(proceeding):
    """
    Determine bond status from proceeding text.
    
    Args:
        proceeding (str): Text from legal proceeding
        
    Returns:
        str: 'BOND_SET', 'BOND_NOT_SET', or 'NONE'
    """
    if not isinstance(proceeding, str):
        return 'NONE'
    
    if BOND_SET_REGEX.search(proceeding):
        return 'BOND_SET'
    elif NO_BOND_SET_REGEX.search(proceeding):
        return 'BOND_NOT_SET'
    else:
        return 'NONE'
    
# SECTION The Court Appointed the Orleans Public Defender

## Pattern
opd_pattern = r"""
(?:
    # Standard appointment patterns with OPD
    \bAPPOINT(?:ED|S|ING)?\s*(?:\S+\s*){0,6}O\.?P\.?D\.?\b
    |
    # Court appointed public defender variations
    \bCOURT\s*APPOINT(?:ED|S|ING)?\s*(?:THE\s*)?(?:ORLEANS\s*)?(?:PARISH\'?S?\s*)?PUBLIC\s*DEFENDER\'?S?\s*(?:OFFICE)?
    |
    # OPD is/was appointed
    \bO\.?P\.?D\.?\s*(?:IS|WAS|ARE|WERE)?\s*APPOINTED
    |
    # Defendant payment to OPD
    \bDEFENDANT\s*TO\s*PAY\s*[$]40\.00\s*TO\s*O\.?P\.?D\.?
    |
    # Court appointed IDP
    \bCOURT\s*APPOINT(?:ED|S|ING)?\s*I\.?D\.?P\.?
    |
    # Public defender office appointed
    \b(?:THE\s*)?(?:ORLEANS\s*)?(?:PARISH\'?S?\s*)?PUBLIC\s*DEFENDER\'?S?\s*(?:OFFICE\s*)?(?:IS|WAS|ARE|WERE)?\s*APPOINTED
)
"""
opd_compiled = re.compile(opd_pattern, flags = re.VERBOSE | re.IGNORECASE)

## Extractor
def extract_the_court_appointed_the_orleans_public_defender(text):
    if text is None or not isinstance(text, str) or text.strip() == "":
        return "NONE"
    try:
        match = opd_compiled.search(text)
        if match:
            return "THE_COURT_APPOINTED_THE_ORLEANS_PUBLIC_DEFENDER"
        else:
            return "NONE"
    except Exception as e:
        return "NONE"

# SECTION Ankle Monitoring Imposed

## Pattern
ankle_patterns = r"""
(?:
    # Fitted/adorned/outfitted/released/placed/wear patterns with ankle monitor
    (?:FIT(?:TED|ING|S)?|ADORN(?:ED|ING|S)?|OUTFIT(?:TED|ING|S)?|RELEAS(?:ED|ING|ES|E)|PLAC(?:ED|ING|ES|E)|WEAR|WORE)\s*
    (?:WITH\s*|ON\s*|W/\s*|FOR\s*)?
    (?:A\s*|AN\s*)?
    (?:G\.?P\.?S\.?\s*ANKLE|ANKLE)\s*MONITOR
    |
    # Enrollment/participation in ankle monitoring program
    (?:ENROLL(?:ING|S)?|PARTICIPAT(?:E|ING|ES))\s*IN\s*AN\s*ANKLE\s*MONITORING\s*PROGRAM
    |
    HAS\s*ENROLLED\s*IN\s*AN\s*ANKLE\s*MONITORING\s*PROGRAM
    |
    # To be monitored through ankle monitor
    TO\s*BE\s*MONITORED\s*THROUGH\s*(?:A\s*|AN\s*)?ANKLE\s*MONITOR(?:ING)?\s*(?:PROGRAM)?
    |
    # Defendant ankle monitor patterns
    DEFENDANT\s*TO\s*(?:WEAR\s*|HAVE\s*)(?:A\s*|AN\s*)?(?:G\.?P\.?S\.?\s*ANKLE|ANKLE)\s*MONITOR
    |
    DEFENDANT\s*(?:MUST\s*)?BE\s*ON\s*(?:A\s*|AN\s*)?(?:G\.?P\.?S\.?\s*ANKLE|ANKLE)\s*MONITOR
    |
    # Ankle monitor as condition of bond
    ANKLE\s*MONITOR(?:\s*(?:AND|WITH|W/)\s*A\s*KEY\s*FOB(?:\s*PROVIDED\s*TO\s*THE\s*VICTIM)?)?\s*IS\s*A\s*CONDITION\s*OF\s*(?:ANY\s*)?BOND
    |
    # Electronic monitoring program
    ELECTRONIC\s*MONITORING\s*(?:PROGRAM\s*)?\(\s*ANKLE\s*BRACELET\s*\)
    |
    # Conditions of bond with ankle monitor
    CONDITIONS\s*OF\s*BOND\s*\:\s*(?:\-)?(?:A\s*|AN\s*)?(?:G\.?P\.?S\.?\s*ANKLE|ANKLE)\s*MONITOR
    |
    # Simple ankle monitor
    \-\s*ANKLE\s*MONITOR
    |
    # Ordered ankle monitor
    ORDER(?:ED|ING|S)?\s*(?:A\s*|AN\s*)?(?:G\.?P\.?S\.?\s*)?ANKLE\s*MONITOR(?:ING)?
    |
    # Ankle monitor placement before release
    ANKLE\s*MONITOR\s*\-\s*WHICH\s*MUST\s*BE\s*PLACED\s*(?:(?:AT\s*OJC|ON\s*(?:THE\s*)?DEFENDANT)\s*)?BEFORE\s*(?:(?:THE\s*)?DEFENDANT(?:\s*IS|\s*\'S)?\s*)?RELEASE
    |
    # Conditions/bonds numbered list
    (?:CONDITIONS|BONDS)\s*\:\s*1\)\s*(?:A\s*|AN\s*)?(?:G\.?P\.?S\.?\s*)?ANKLE\s*MONITOR(?:ING)?
    |
    # Be on ankle monitor
    BE\s*ON\s*(?:A\s*|AN\s*)?(?:G\.?P\.?S\.?\s*)?ANKLE\s*MONITOR(?:ING)?
    |
    # Court ordered ankle monitor
    COURT\s*ORDERED?\s*(?:THE\s*)?DEFENDANT\s*(?:TO\s*)?.*?ANKLE\s*(?:MONITOR|MONITORING)
    |
    # Bond with house arrest and ankle monitor
    (?:CONDITION\s*OF\s*)?BOND\s*:\s*HOUSE\s*(?:A\s*/\s*R|ARREST)\s*(?:AND|WITH)\s*ANKLE\s*(?:MONITOR|MONITORING)
    |
    # Defendant ordered to ankle GPS
    DEFENDANT\s*(?:IS\s*)?ORDERED\s*TO\s*ANKLE\s*GPS\s*(?:MONITOR|MONITORING)
    |
    # State's request for GPS ankle monitor
    (?:GRANTED\s*)?(?:THE\s*)?STATE'?S\s*REQUEST\s*FOR\s*(?:A\s*)?GPS\s*ANKLE\s*MONITOR
    |
    # Placed on 24-hour ankle monitoring
    (?:BE\s*)?PLACED\s*ON\s*(?:24\s*-?\s*HOUR\s*)?ANKLE\s*(?:MONITOR|MONITORING)
    |
    # Ankle monitor as condition
    (?:AN\s*)?ANKLE\s*MONITOR\s*IS\s*A\s*CONDITION\s*(?:OF\s*(?:BOND|RELEASE))?
    |
    # 24-hour ankle monitoring upon posting bond
    24\s*-?\s*HOUR\s*ANKLE\s*(?:MONITOR|MONITORING)\s*UPON\s*POSTING\s*BOND
)
"""
ankle_compiled = re.compile(ankle_patterns, flags = re.VERBOSE | re.IGNORECASE)
## Extractor
def extract_ankle_monitoring(text):
    if text is None or not isinstance(text, str) or text.strip() == "":
        return "NONE"
    try:
        match = ankle_compiled.search(text)
        if match:
            return "ANKLE_MONITORING_IMPOSED"
        else:
            return "NONE"
    except Exception as e:
        return "NONE"

# SECTION Court Ordered Domestic Violence Stay Away

## Pattern
domestic_violence_order_patterns = r"""
(?:
    # Court issued/ordered/executed domestic violence stay away order
    COURT\s*(?:ALSO\s*)?
    (?:ISSU(?:E|ED|ES|ING)|ORDER(?:ED|S|ING)|EXECUT(?:E|ED|ES|ING))\s*
    (?:A|AN)?
    (?:
        # Numbered patterns with parentheses: (1), (2), etc.
        \(\s*(\d+)\s*\)\s*
        |
        # Direct number: A 1, AN 2, 3, etc.
        (?:A|AN)?(\d+)\s*
        |
        # Ordinal numbers: 1ST, 2ND, 3RD, 4TH, etc.
        (?:A|AN)?\s*(\d+)(?:ST|ND|RD|TH)\s*
        |
        # Written numbers with ordinal suffixes
        (?:A|AN)?\b
        (?:
            (?:ZERO|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|NINETEEN|TWENTY|THIRTY|FORTY|FIFTY|SIXTY|SEVENTY|EIGHTY|NINETY|HUNDRED|THOUSAND|MILLION|BILLION|TRILLION)
            (?:\s+AND)?\s*
        )
        (?:FIR|SECO|THI|FOUR|FIF|SIX|SEVEN|EIGH|NIN|TEN|ELEVEN|TWELF|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|NINETEEN|TWENTIE|THIRTIE|FORTIE|FIFTIE|SIXTIE|SEVENTIE|EIGHTIE|NINETIE|HUNDRED|THOUSAND|MILLION|BILLION|TRILLION)
        (?:ST|ND|RD|TH)\s*
        |
        # Multiple written numbers
        (?:A|AN)?
        (?:
            \b(?:
                (?:ZERO|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|NINETEEN|TWENTY|THIRTY|FORTY|FIFTY|SIXTY|SEVENTY|EIGHTY|NINETY|HUNDRED|THOUSAND|MILLION|BILLION|TRILLION)
                (?:\s+AND)?\s*
            )+\b
        )
    )?
    (?:\s*VERBAL\s*|\s*ORAL\s*)?\s*
    DOMESTIC\s*(?:VIOLENCE\s*)?
    (?:STAY\s*AWAY(?:S\s*|\s*ORDERS?)?|S\.?A\.?O\.?)
    |
    # Court granted motion for domestic violence stay away order
    COURT\s*
    GRANT(?:ED|S|ING)?\s*
    (?:THE\s*)?
    (?:STATE\'?S\s*)?
    MOTION\s*
    (?:FOR\s*)?
    DOMESTIC\s*(?:VIOLENCE\s*)?
    (?:STAY\s*AWAY(?:S\s*|\s*ORDERS?)?|S\.?A\.?O\.?)
    |
    #defendant completed domestic stay away order
    DEFENDANT\s*
    COMPLETED\s*
    (?:A\s*|AN\s*)?
    DOMESTIC\s*(?:VIOLENCE\s*)?
    (?:STAY\s*AWAY(?:S\s*|\s*ORDERS?)?|S\.?A\.?O\.?)
)
"""
domestic_violence_compiled = re.compile(domestic_violence_order_patterns, flags = re.VERBOSE | re.IGNORECASE)

## Extractor
def extract_domestic_stay_away_order(text):
    if text is None or not isinstance(text, str) or text.strip() == "":
        return "NONE"
    try:
        match = domestic_violence_compiled.search(text)
        if match:
            return "DOMESTIC_STAY_AWAY_ORDER_IMPOSED"
        else:
            return "NONE"
    except Exception as e:
        return "NONE"

# SECTION Charge Status

## Patterns

### Accepted Charges
accepted_charges_pattern = r"""
(?ix)
(?:
    # DA/STATE accepting patterns
    (?:STATE|THE\s+DA|DA|DISTRICT\s+ATTORNEY)\s+
    (?:ACCEPTED?|ACCEPTS?|ACC\.?|\s+OTHER\s+|\s+WILL\s+(?:ACCEPT|PROSECUTE)|MOVED\s+FORWARD|PROCEEDING|TAKES?\s+UP|PROCEEDS?)\s+
    (?:\s+)?(?:CHARGE[S()]?|PROSECUTION)?(?:\s+TO)?
    |
    # Charges accepted by patterns
    CHARGES?\s+(?:ACCEPTED|TAKEN\s+UP)\s+BY\s+(?:STATE|DISTRICT\s+ATTORNEY)
    |
    # Prosecution accepted patterns
    PROSECUTION\s+ACCEPTED\s+BY\s+THE\s+STATE
    |
    # Simple ACC pattern
    ACC\.
)
"""
ACCEPTED_CHARGES_REGEX = re.compile(accepted_charges_pattern, re.VERBOSE | re.IGNORECASE)

### Refused Charges
refused_charges_pattern = r"""
(?ix)
(?:
    # DA/STATE refusing patterns
    (?:STATE|DISTRICT\s+ATTORNEY|PROSECUTION|THE\s+STATE)\s+
    (?:\s+)?
    (?:REFUSED|DECLINED|DROP(?:PED)?|DID\s+NOT\s+ACCEPT|DECLINES)\s+
    (?:\s+PROSECUTE|(?:ALL|REMAINING|PENDING|OTHER|CHARGE[S()]|\w+))*
    |
    # Charge disposal patterns
    (?:CHARGE[S()]|THIS\s+MATTER|MATTER|CASE)\s+
    (?:DISPOSED\s+OF|(?:\s+BEEN\s+)?REFUSED|DROPPED|DECLINED\s+FOR\s+PROSECUTION)
    |
    # Refused with date patterns
    REFUSED\s+[\w\s]*(?:ON|N)\s+[\w\s]+
    |
    # Simple refused/refuse patterns
    REFUSED\s+[\w\s]+|REFUSE\s+[\w\s]+|[\w\s]+REFUSE[D]?
    |
    # Nolle prosequi patterns
    ENTERED\s+A\s+NOLLE\s+PROS(?:EQUI)?(?:\s+[\w\s]+)?
    |
    NOLLE\s+PROS(?:EQUI)?(?:ED)?(?:\s+[\w\s]+)?
    |
    ORDER\s+OF\s+NOLLE\s+PROS(?:EUI)?\s+ISSUED(?:\s+[\w\s]+)?
    |
    THIS\s+MATTER\s+HAS\s+BEEN\s+NOLLE\s+PROSEQUI
    |
    # DA REF patterns
    DA\s+REFUSED\s+(?:ALL|OTHER|CHARGES?|[\w\s]+)?
    |
    # Other declined prosecution patterns
    PROSECUTION\s+DECLINED
)
"""
REFUSED_CHARGES_REGEX = re.compile(refused_charges_pattern, re.VERBOSE | re.IGNORECASE)

### Mixed Charges
mixed_charges_pattern = r"""
(?ix)
(?:
    # Accept and refuse patterns
    ACCEPT\s+[\w\s:,.()-]+\s+(?:ALL|BUT|WHILE)\s+(?:OTHERS?)\s+REFUSED
    |
    REFUSE\s+[\w\s:,.()-]+\s+AND\s+ACCEPT\s+[\w\s:,.()-]+
    |
    ACCEPT\s+[\w\s:,.()-]+\s*-\s*ALL\s+OTHERS\s+REFUSED
    |
    # DA/STATE accepted and refused patterns
    (?:STATE)\s+ACCEPTED\s+[\w\s:,.()-]+(?:,)?\s+REFUSED\s+[\w\s:,.()-]+
    |
    #Accepted and refused patterns with exclusions for certain phrases such for the cases of defendant, pending, to follow, not yet been, all charges have been, the case was not, status hearing set to see if charges were, and date phrases
    (?<!DEFENDANT\s*)                                                    
    (?<!PENDING\s*)                                                      
    (?<!TO\s*FOLLOW\s*CASE\s*\d+\s*WHICH\s*WAS\s*)                      
    (?<!THE\s*(?:CHARGES\s*HAVE|CASE\s*HAS)\s*NOT\s*(?:YET\s*)?BEEN?\s*) 
    (?<!ALL\s*CHARGES\s*HAVE\s*BEEN?\s*)                                
    (?<!THE\s*CASE\s*WAS\s*NOT\s*)                                       
    (?<!STATUS\s*HEARING\s*SET\s*TO\s*SEE\s*IF\s*CHARGES\s*WERE\s*)      
    ACCEPTED\s+[\w\s:,.()-]+(?:,)?\s+                                    
    (?<!DEFENDANT\s)                                                     
    (?<!PENDING\s)                                                       
    (?<!TO\s*FOLLOW\s*CASE\s*\d+\s*WHICH\s*WAS\s*)                      
    REFUSED\s+(?:(?!.*(?:PRIOR\s*TO\s*TODAY(?:\'?S)?\s*|BEFORE\s*(?:THIS\s*|TODAY(?:\'?S)?\s*)?)(?:COURT\s*)?DATE)[\w\s:,.()-])+
    |
    # Charges with numbers patterns
    CHARGES?\s+[\d-]+\s+ACCEPTED(?:,)?\s+[\d-]+\s+REFUSED
    |
    # Prosecute some, decline others patterns
    (?:STATE)\s+(?:\s+)?PROSECUTE\s+[\w\s:,.()-]+(?:,)?\s+(?:REST)\s+(?:DISPOSED|REFUSED)
    |
    # Accept but refuse patterns
    (?:STATE)\s+ACCEPTS\s+[\w\s:,.()-]+\s+(?:WHILE)\s+REFUSES\s+[\w\s:,.()-]+
    |
    # Proceeding with some, declining others patterns
    PROCEEDING\s+WITH\s+[\w\s:,.()-]+(?:,)?\s+DECLINING\s+[\w\s:,.()-]+
    |
    # DA/STATE accepted and refused general pattern
    (?:STATE)\s+ACCEPTED\s+[\w\s:,.()-]+(?:,)?\s+REFUSED\s+[\w\s:,.()-]+
)
"""
MIXED_CHARGES_REGEX = regex.compile(mixed_charges_pattern, regex.IGNORECASE | regex.VERBOSE | regex.MULTILINE)

## Extractor
def determine_charge_status(proceeding):
  if not isinstance(proceeding, str):
    return "NONE"
  if MIXED_CHARGES_REGEX.search(proceeding) or ACCEPTED_CHARGES_REGEX.search(proceeding):
    return 'CHARGES_ACCEPTED'
  elif REFUSED_CHARGES_REGEX.search(proceeding) and not ACCEPTED_CHARGES_REGEX.search(proceeding):
    return 'CHARGES_REFUSED'
  else:
    return 'NO_DECISION'

# SECTION Gwen's Law Hearing

# ## Pattern
# gwens_law_compiled = re.compile(r'^\s*GWEN(?:(?:\')?S)?\s*LAW\s*HEARING\s*IN\s*MAGISTRATE\s*COURT\s*SECTION', re.MULTILINE)

# ## Extractor
# def extract_gwens_law_hearing(text):
#     if text is None or not isinstance(text, str) or text.strip() == "":
#         return "NONE"
#     try:
#         match = gwens_law_compiled.search(text)
#         if match:
#             return "GWENS_LAW_HEARING_OCCURRED"
#         else:
#             return "NONE"
#     except Exception as e:
#         return "NONE"

# SECTION Probable Cause Status

## Patterns

### Probable Cause Found
probable_cause_pattern = r"""
(?ix)
(?:
    # Abbreviation patterns
    \b(?<!NO\s)(?<!NO)(?<!N\s)(?<!NP\s)(?<!NP)(?:P\.C\.?|P/C|PC|PRC|PCF|PRCF|P\.C\.F\.?|PRC\.F\.?)\b
    |
    # Standard patterns with "found", "exists", etc.
    \b(?<!NO\s)(?<!NO)(?<!N\s)(?<!N)(?:P\.C\.?|P/C|PC|PRC|PROB(?:ABLE)?\s+CAUSE)(?:\s+(?:WAS\s+)?(?:FOUND|EXISTS|FOR))?
    |
    # The TCF prefixed patterns
    \bTCF(?:PC|PRC)(?!N)\b
    |
    \b(?<!NO\s)(?<!NOT\s)(?:THE\s+)?COURT(?:\s+(?:HAS\s+)?FOUND|\s+FINDS)\s+PROB(?:ABLE)?\s+CAUSE\b
)
"""
PROBABLE_CAUSE_REGEX = re.compile(probable_cause_pattern, re.VERBOSE | re.IGNORECASE)

### No Probable Cause Found
no_probable_cause_pattern = r"""
(?ix)
(?:
    # Standard no patterns
    \bN(?:O|P)\s+(?:P\.C\.?|P/C|PC|PROB(?:ABLE)?\s+CAUSE)(?:\s+(?:WAS\s+)?(?:FOUND|EXISTS|FOR))?
    |
    # No abbreviation patterns
    \b(?:NPC|NPRC|NPCF|NPRCF)\b
    |
    # TCF patterns
    \bTCFNP(?:C|RC)\b
)
"""
NO_PROBABLE_CAUSE_REGEX = re.compile(no_probable_cause_pattern, re.VERBOSE | re.IGNORECASE)

## Extractor
def has_probable_cause(proceeding):
  if PROBABLE_CAUSE_REGEX.search(proceeding):
    return 'PROBABLE_CAUSE_FOUND'
  elif NO_PROBABLE_CAUSE_REGEX.search(proceeding):
    return 'PROBABLE_CAUSE_NOT_FOUND'
  else:
    return 'NONE'

# SECTION In Custody and Not In Court Status

## Patterns
### Defendant In Custody
in_custody_pattern = r"""
(?:
    # PDOJL abbreviation
    PDOJL
    |
    # Put defendant on jail list
    PUT\s*DEFENDANT\s*ON\s*JAIL\s*LIST
    |
    # In custody and not brought to court variations
    (?:IS\s*|WAS\s*)?(?:IN\s*)?(?:THE\s*)?CUSTODY\s*AND\s*
    (?:IS\s*|WAS\s*)?
    (?:
        NOT\s*(?:BROUGHT|TRANSPORTED|TRANSFERRED)
        |
        (?:REFUSED\s*|UNABLE\s*|FAILED\s*)
        (?:
            TO\s*(?:BE\s*(?:BROUGHT|TRANSPORTED)|APPEAR|COME|BROUGHT|LEAVE)
            |
            TRANSPORTED
        )
        |
        (?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE|BROUGHT(?:\s*OVER)?|TRANSPORTED(?:\s*OVER)?)|UNAVAILABLE|ABSENT)
        |
        DID\s*NOT\s*(?:APPEAR|COME|MAKE\s*AN\s*APPEARANCE)
        |
        NOT\s*AVAILABLE\s*THIS\s*DAY
    )
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    (?:\s*
    (?:(?:TO|INTO|IN\s*TO|UP(?:\s*FOR|\s*IN|\s*TO)?)\s*)?
    (?:
        (?:ZOOM\s*OR\s*)?OPEN\s*COURT
        |
        OPEN/ZOOM\s*COURT
        |
        ZOOM\s*(?:COURT|HEARING|MEETING)
        |
        COURT
        |
        TO\s*VIRTUAL\s*COURTROOM
    ))?
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    |
    # In custody of specific facilities and not brought
    (?:IS|WAS)\s*IN\s*
    (?:
        THE\s*CUSTODY\s*OF\s*(?:THE\s*)?(?:{all_louisiana_jails_pattern})
        |
        FEDERAL\s*CUSTODY
    )\s*
    AND\s*(?:IS\s*|WAS\s*)?
    (?:
        NOT\s*(?:BROUGHT|TRANSPORTED|TRANSFERRED)
        |
        (?:REFUSED\s*|UNABLE\s*|FAILED\s*)
        (?:
            TO\s*(?:BE\s*(?:BROUGHT|TRANSPORTED)|APPEAR|COME|BROUGHT|LEAVE)
            |
            TRANSPORTED
        )
        |
        (?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE|BROUGHT(?:\s*OVER)?|TRANSPORTED(?:\s*OVER)?)|UNAVAILABLE|ABSENT)
        |
        DID\s*NOT\s*(?:APPEAR|COME|MAKE\s*AN\s*APPEARANCE)
        |
        NOT\s*AVAILABLE\s*THIS\s*DAY
    )
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    (?:\s*
    (?:(?:TO|INTO|IN\s*TO|UP(?:\s*FOR|\s*IN|\s*TO)?)\s*)?
    (?:
        (?:ZOOM\s*OR\s*)?OPEN\s*COURT
        |
        OPEN/ZOOM\s*COURT
        |
        ZOOM\s*(?:COURT|HEARING|MEETING)
        |
        COURT
        |
        TO\s*VIRTUAL\s*COURTROOM
    ))?
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    |
    # Facility custody with general pattern
    (?:IS|WAS)\s*IN\s*(?:(?:THE\s*)?(?:{all_louisiana_jails_pattern})\s*CUSTODY)\s*
    AND\s*(?:IS\s*|WAS\s*)?
    (?:
        NOT\s*(?:BROUGHT|TRANSPORTED|TRANSFERRED)
        |
        (?:REFUSED\s*|UNABLE\s*|FAILED\s*)
        (?:
            TO\s*(?:BE\s*(?:BROUGHT|TRANSPORTED)|APPEAR|COME|BROUGHT|LEAVE)
            |
            TRANSPORTED
        )
        |
        (?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE|BROUGHT(?:\s*OVER)?|TRANSPORTED(?:\s*OVER)?)|UNAVAILABLE|ABSENT)
        |
        DID\s*NOT\s*(?:APPEAR|COME|MAKE\s*AN\s*APPEARANCE)
        |
        NOT\s*AVAILABLE\s*THIS\s*DAY
    )
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?    
    (?:\s*
    (?:(?:TO|INTO|IN\s*TO|UP(?:\s*FOR|\s*IN|\s*TO)?)\s*)?
    (?:
        (?:ZOOM\s*OR\s*)?OPEN\s*COURT
        |
        OPEN/ZOOM\s*COURT
        |
        ZOOM\s*(?:COURT|HEARING|MEETING)
        |
        COURT
        |
        TO\s*VIRTUAL\s*COURTROOM
    ))?
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    |
    # Parish custody patterns
    (?:IS|WAS)\s*IN\s*(?:THE\s*)?(?:FEDERAL\s*)?CUSTODY\s*(?:OF|IN)\s*
    (?:THE\s*)?(?:SHERIFF\s*(?:IN\s*)?)?(?:DOC/\s*)?
    (?:{louisiana_parishes_pattern}|ANOTHER)
    (?:\s*PARISH)?(?:\s*SHERIFF\'?S?|\s*JAIL)?\s*
    AND\s*(?:IS\s*|WAS\s*)?
    (?:
        NOT\s*(?:BROUGHT|TRANSPORTED|TRANSFERRED)
        |
        (?:REFUSED\s*|UNABLE\s*|FAILED\s*)
        (?:
            TO\s*(?:BE\s*(?:BROUGHT|TRANSPORTED)|APPEAR|COME|BROUGHT|LEAVE)
            |
            TRANSPORTED
        )
        |
        (?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE|BROUGHT(?:\s*OVER)?|TRANSPORTED(?:\s*OVER)?)|UNAVAILABLE|ABSENT)
        |
        DID\s*NOT\s*(?:APPEAR|COME|MAKE\s*AN\s*APPEARANCE)
        |
        NOT\s*AVAILABLE\s*THIS\s*DAY
    )
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?    
    (?:\s*
    (?:(?:TO|INTO|IN\s*TO|UP(?:\s*FOR|\s*IN|\s*TO)?)\s*)?
    (?:
        (?:ZOOM\s*OR\s*)?OPEN\s*COURT
        |
        OPEN/ZOOM\s*COURT
        |
        ZOOM\s*(?:COURT|HEARING|MEETING)
        |
        COURT
        |
        TO\s*VIRTUAL\s*COURTROOM
    ))?
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    |
    # Regular sheriff custody patterns
    (?:IS|WAS)\s*IN\s*(?:THE\s*)?(?:FEDERAL\s*)?CUSTODY\s*(?:OF|IN)\s*
    (?:THE\s*)?(?:SHERIFF\s*)
    AND\s*(?:IS\s*|WAS\s*)?
    (?:
        NOT\s*(?:BROUGHT|TRANSPORTED|TRANSFERRED)
        |
        (?:REFUSED\s*|UNABLE\s*|FAILED\s*)
        (?:
            TO\s*(?:BE\s*(?:BROUGHT|TRANSPORTED)|APPEAR|COME|BROUGHT|LEAVE)
            |
            TRANSPORTED
        )
        |
        (?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE|BROUGHT(?:\s*OVER)?|TRANSPORTED(?:\s*OVER)?)|UNAVAILABLE|ABSENT)
        |
        DID\s*NOT\s*(?:APPEAR|COME|MAKE\s*AN\s*APPEARANCE)
        |
        NOT\s*AVAILABLE\s*THIS\s*DAY
    )
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY'?S\s*COURT\s*SETTING|TO\s*COME))?    
    (?:\s*
    (?:(?:TO|INTO|IN\s*TO|UP(?:\s*FOR|\s*IN|\s*TO)?)\s*)?
    (?:
        (?:ZOOM\s*OR\s*)?OPEN\s*COURT
        |
        OPEN/ZOOM\s*COURT
        |
        ZOOM\s*(?:COURT|HEARING|MEETING)
        |
        COURT
        |
        TO\s*VIRTUAL\s*COURTROOM
    ))?
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY'?S\s*COURT\s*SETTING|TO\s*COME))?
    |
    # Federal detention facility patterns
    (?:IS|WAS)\s*IN\s*(?:THE\s*)?(?:SHERIFF\s*(?:IN\s*)?)?(?:DOC/\s*)?
    (?:{louisiana_parishes_pattern}|ANOTHER)
    (?:\s*PARISH)?(?:\s*SHERIFF\'?S?|\s*JAIL)?\s*
    (?:
        (?:FEDERAL\s*)?CUSTODY\s*(?:OF|IN)\s*(?:THE\s*)?(?:FEDERAL\s*)?
        (?:
            DETENTION\s*CENTER
            |
            BUREAU\s*OF\s*PRISONS
            |
            BOP|FDC|MCC|FCI|USP
            |
            MARSHALS?\s*SERVICE
            |
            ICE|IMMIGRATION
        )?
    )?\s*
    AND\s*(?:IS\s*|WAS\s*)?
    (?:
        NOT\s*(?:BROUGHT|TRANSPORTED|TRANSFERRED)
        |
        (?:REFUSED\s*|UNABLE\s*|FAILED\s*)
        (?:
            TO\s*(?:BE\s*(?:BROUGHT|TRANSPORTED)|APPEAR|COME|BROUGHT|LEAVE)
            |
            TRANSPORTED
        )
        |
        (?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE|BROUGHT(?:\s*OVER)?|TRANSPORTED(?:\s*OVER)?)|UNAVAILABLE|ABSENT)
        |
        DID\s*NOT\s*(?:APPEAR|COME|MAKE\s*AN\s*APPEARANCE)
        |
        NOT\s*AVAILABLE\s*THIS\s*DAY
    )
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?    
    (?:\s*
    (?:(?:TO|INTO|IN\s*TO|UP(?:\s*FOR|\s*IN|\s*TO)?)\s*)?
    (?:
        (?:ZOOM\s*OR\s*)?OPEN\s*COURT
        |
        OPEN/ZOOM\s*COURT
        |
        ZOOM\s*(?:COURT|HEARING|MEETING)
        |
        COURT
        |
        TO\s*VIRTUAL\s*COURTROOM
    ))?
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    |
    # General custody without court appearance
    IN\s*(?:THE\s*)?CUSTODY\s*AND\s*
    (?:IS\s*|WAS\s*)?
    (?:
        NOT\s*(?:BROUGHT|TRANSPORTED|TRANSFERRED)
        |
        (?:REFUSED\s*|UNABLE\s*|FAILED\s*)
        (?:
            TO\s*(?:BE\s*(?:BROUGHT|TRANSPORTED)|APPEAR|COME|BROUGHT|LEAVE)
            |
            TRANSPORTED
        )
        |
        (?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE|BROUGHT(?:\s*OVER)?|TRANSPORTED(?:\s*OVER)?)|UNAVAILABLE|ABSENT)
        |
        DID\s*NOT\s*(?:APPEAR|COME|MAKE\s*AN\s*APPEARANCE)
        |
        NOT\s*AVAILABLE\s*THIS\s*DAY
    )
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    |
    # Defendant/inmate in custody and housed at facility
    (?:DEFENDANT|INMATE)\s+
    (?:IS\s+|WAS\s+|HAS\s+BEEN\s+|REMAINS\s+)?
    (?:IN\s+)?(?:(?:FEDERAL\s+)?CUSTODY\s+)?
    (?:AND\s+)?
    (?:
        HOUSED\s+(?:AT|IN)
        |
        HELD\s+(?:AT|IN)
        |
        INCARCERATED\s+IN
        |
        ARRESTED\s+AND\s+HELD\s+IN
        |
        NOT\s+(?:TRANSPORTED|BROUGHT)\s+FROM
        |
        (?:THE\s+)?CUSTODY\s+OF
        |
        JAIL\s+IN
        |
        CURRENTLY\s+HELD\s+IN
    )\s*
    (?:{all_louisiana_jails_pattern})
    |
    # Special status conditions and not transported/did not appear
    (?:THE\s*)?DEFENDANT\s*(?:IS\s*|WAS\s*)?
    (?:STILL\s*|REMAINS\s*|CURRENTLY\s*)?
    (?:
        (?:IN|ON|IS|WAS)\s*(?:THE\s*|THIS\s*)?(?:DEXTOX|DETOX|HOSPITAL|JAIL|MEDICAL)
        |
        (?:IN|ON|IS|WAS)\s*(?:SUICIDE\s*WATCH|LOCKDOWN|ISOLATION)
        |
        (?:IN|ON|IS|WAS)\s*(?:THE\s*|THIS\s*)?(?:(?:MAX\.?\s*SECURITY|MEDICAL)\s*TIER)
        |
        A\s*SECURITY\s*ISSUE
        |
        (?:DETOXIFYING|DETOXYING)
        |
        (?:BEING\s*)?COMBATIVE
    )\s*
    (?:(?:@|AT)\s*(?:{all_louisiana_jails_pattern})\s*)?
    (?:
        AND\s*(?:IS\s*|WAS\s*)?
        (?:
            NOT\s*(?:BROUGHT|TRANSPORTED|TRANSFERRED)
            |
            (?:REFUSED\s*|UNABLE\s*|FAILED\s*)
            (?:
                TO\s*(?:BE\s*(?:BROUGHT|TRANSPORTED)|APPEAR|COME|BROUGHT|LEAVE)
                |
                TRANSPORTED
            )
            |
            (?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE|BROUGHT(?:\s*OVER)?|TRANSPORTED(?:\s*OVER)?)|UNAVAILABLE|ABSENT)
            |
            DID\s*NOT\s*(?:APPEAR|COME|MAKE\s*AN\s*APPEARANCE)
        )
        (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    )?
    (?:\s*
        (?:(?:TO|INTO|IN\s*TO|UP(?:\s*FOR|\s*IN|\s*TO)?)\s*)?
        (?:
            (?:ZOOM\s*OR\s*)?OPEN\s*COURT
            |
            OPEN/ZOOM\s*COURT
            |
            ZOOM\s*(?:COURT|HEARING|MEETING)
            |
            COURT
            |
            TO\s*VIRTUAL\s*COURTROOM
        )
    )?
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    |
    ^NOT\s*(?:BROUGHT|TRANSPORTED)(?:\s*TO\s*(?:COURT|THIS\s*COURT\s*SETTING|TODAY\'S\s*COURT\s*SETTING|TODAY|VIRTUAL\s*COURTROOM|VIRTUAL\s*HEARING|VIRTUAL\s*MEETING|ZOOM\s*COURT|OPEN/ZOOM\s*COURT|OPEN\s*COURT|ZOOM\s*OR\s*OPEN\s*COURT))?
)
"""

### Defendant Not In Court
not_in_court_pattern = r"""
(?:
    # Defendant not brought/transported to court
    DEFENDANT\s*(?:IS\s*|WAS\s*)?
    (?:
        NOT\s*(?:BROUGHT|TRANSPORTED|TRANSFERRED)
        |
        (?:REFUSED\s*|UNABLE\s*|FAILED\s*)
        (?:
            TO\s*(?:BE\s*(?:BROUGHT|TRANSPORTED)|APPEAR|COME|BROUGHT|LEAVE)
            |
            TRANSPORTED
        )
        |
        (?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE|BROUGHT(?:\s*OVER)?|TRANSPORTED(?:\s*OVER)?)|UNAVAILABLE|ABSENT)
        |
        DID\s*NOT\s*(?:APPEAR|COME|MAKE\s*AN\s*APPEARANCE)
        |
        NOT\s*AVAILABLE\s*THIS\s*DAY
    )
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    (?:\s*
    (?:(?:TO|INTO|IN\s*TO|UP(?:\s*FOR|\s*IN|\s*TO)?)\s*)?
    (?:
        (?:ZOOM\s*OR\s*)?OPEN\s*COURT
        |
        OPEN/ZOOM\s*COURT
        |
        ZOOM\s*(?:COURT|HEARING|MEETING)
        |
        COURT
        |
        TO\s*VIRTUAL\s*COURTROOM
    ))?
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    |
    # In custody and not brought variations
    (?:IS\s*|WAS\s*)?(?:IN\s*)?(?:THE\s*)?CUSTODY\s*AND\s*
    (?:IS\s*|WAS\s*)?
    (?:
        NOT\s*(?:BROUGHT|TRANSPORTED|TRANSFERRED)
        |
        (?:REFUSED\s*|UNABLE\s*|FAILED\s*)
        (?:
            TO\s*(?:BE\s*(?:BROUGHT|TRANSPORTED)|APPEAR|COME|BROUGHT|LEAVE)
            |
            TRANSPORTED
        )
        |
        (?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE|BROUGHT(?:\s*OVER)?|TRANSPORTED(?:\s*OVER)?)|UNAVAILABLE|ABSENT)
        |
        DID\s*NOT\s*(?:APPEAR|COME|MAKE\s*AN\s*APPEARANCE)
        |
        NOT\s*AVAILABLE\s*THIS\s*DAY
    )
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    (?:\s*
    (?:(?:TO|INTO|IN\s*TO|UP(?:\s*FOR|\s*IN|\s*TO)?)\s*)?
    (?:
        (?:ZOOM\s*OR\s*)?OPEN\s*COURT
        |
        OPEN/ZOOM\s*COURT
        |
        ZOOM\s*(?:COURT|HEARING|MEETING)
        |
        COURT
        |
        TO\s*VIRTUAL\s*COURTROOM
    ))?
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    |
    # Specific facility custody patterns
    (?:IS|WAS)\s*IN\s*
    (?:
        THE\s*CUSTODY\s*OF\s*(?:THE\s*)?(?:{all_louisiana_jails_pattern})
        |
        FEDERAL\s*CUSTODY
    )\s*
    AND\s*(?:IS\s*|WAS\s*)?
    (?:
        NOT\s*(?:BROUGHT|TRANSPORTED|TRANSFERRED)
        |
        (?:REFUSED\s*|UNABLE\s*|FAILED\s*)
        (?:
            TO\s*(?:BE\s*(?:BROUGHT|TRANSPORTED)|APPEAR|COME|BROUGHT|LEAVE)
            |
            TRANSPORTED
        )
        |
        (?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE|BROUGHT(?:\s*OVER)?|TRANSPORTED(?:\s*OVER)?)|UNAVAILABLE|ABSENT)
        |
        DID\s*NOT\s*(?:APPEAR|COME|MAKE\s*AN\s*APPEARANCE)
        |
        NOT\s*AVAILABLE\s*THIS\s*DAY
    )
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    (?:\s*
    (?:(?:TO|INTO|IN\s*TO|UP(?:\s*FOR|\s*IN|\s*TO)?)\s*)?
    (?:
        (?:ZOOM\s*OR\s*)?OPEN\s*COURT
        |
        OPEN/ZOOM\s*COURT
        |
        ZOOM\s*(?:COURT|HEARING|MEETING)
        |
        COURT
        |
        TO\s*VIRTUAL\s*COURTROOM
    ))?
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    |
    # Facility custody with general pattern
    (?:IS|WAS)\s*IN\s*(?:(?:THE\s*)?(?:{all_louisiana_jails_pattern})\s*CUSTODY)\s*
    AND\s*(?:IS\s*|WAS\s*)?
    (?:
        NOT\s*(?:BROUGHT|TRANSPORTED|TRANSFERRED)
        |
        (?:REFUSED\s*|UNABLE\s*|FAILED\s*)
        (?:
            TO\s*(?:BE\s*(?:BROUGHT|TRANSPORTED)|APPEAR|COME|BROUGHT|LEAVE)
            |
            TRANSPORTED
        )
        |
        (?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE|BROUGHT(?:\s*OVER)?|TRANSPORTED(?:\s*OVER)?)|UNAVAILABLE|ABSENT)
        |
        DID\s*NOT\s*(?:APPEAR|COME|MAKE\s*AN\s*APPEARANCE)
        |
        NOT\s*AVAILABLE\s*THIS\s*DAY
    )
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?    
    (?:\s*
    (?:(?:TO|INTO|IN\s*TO|UP(?:\s*FOR|\s*IN|\s*TO)?)\s*)?
    (?:
        (?:ZOOM\s*OR\s*)?OPEN\s*COURT
        |
        OPEN/ZOOM\s*COURT
        |
        ZOOM\s*(?:COURT|HEARING|MEETING)
        |
        COURT
        |
        TO\s*VIRTUAL\s*COURTROOM
    ))?
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    |
    # Parish custody patterns
    (?:IS|WAS)\s*IN\s*(?:THE\s*)?(?:FEDERAL\s*)?CUSTODY\s*(?:OF|IN)\s*
    (?:THE\s*)?(?:SHERIFF\s*(?:IN\s*)?)?(?:DOC/\s*)?
    (?:{louisiana_parishes_pattern}|ANOTHER)
    (?:\s*PARISH)?(?:\s*SHERIFF\'?S?|\s*JAIL)?\s*
    AND\s*(?:IS\s*|WAS\s*)?
    (?:
        NOT\s*(?:BROUGHT|TRANSPORTED|TRANSFERRED)
        |
        (?:REFUSED\s*|UNABLE\s*|FAILED\s*)
        (?:
            TO\s*(?:BE\s*(?:BROUGHT|TRANSPORTED)|APPEAR|COME|BROUGHT|LEAVE)
            |
            TRANSPORTED
        )
        |
        (?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE|BROUGHT(?:\s*OVER)?|TRANSPORTED(?:\s*OVER)?)|UNAVAILABLE|ABSENT)
        |
        DID\s*NOT\s*(?:APPEAR|COME|MAKE\s*AN\s*APPEARANCE)
        |
        NOT\s*AVAILABLE\s*THIS\s*DAY
    )
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?    
    (?:\s*
    (?:(?:TO|INTO|IN\s*TO|UP(?:\s*FOR|\s*IN|\s*TO)?)\s*)?
    (?:
        (?:ZOOM\s*OR\s*)?OPEN\s*COURT
        |
        OPEN/ZOOM\s*COURT
        |
        ZOOM\s*(?:COURT|HEARING|MEETING)
        |
        COURT
        |
        TO\s*VIRTUAL\s*COURTROOM
    ))?
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    |
    # Regular sheriff custody patterns
    (?:IS|WAS)\s*IN\s*(?:THE\s*)?(?:FEDERAL\s*)?CUSTODY\s*(?:OF|IN)\s*
    (?:THE\s*)?(?:SHERIFF\s*)
    AND\s*(?:IS\s*|WAS\s*)?
    (?:
        NOT\s*(?:BROUGHT|TRANSPORTED|TRANSFERRED)
        |
        (?:REFUSED\s*|UNABLE\s*|FAILED\s*)
        (?:
            TO\s*(?:BE\s*(?:BROUGHT|TRANSPORTED)|APPEAR|COME|BROUGHT|LEAVE)
            |
            TRANSPORTED
        )
        |
        (?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE|BROUGHT(?:\s*OVER)?|TRANSPORTED(?:\s*OVER)?)|UNAVAILABLE|ABSENT)
        |
        DID\s*NOT\s*(?:APPEAR|COME|MAKE\s*AN\s*APPEARANCE)
        |
        NOT\s*AVAILABLE\s*THIS\s*DAY
    )
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY'?S\s*COURT\s*SETTING|TO\s*COME))?    
    (?:\s*
    (?:(?:TO|INTO|IN\s*TO|UP(?:\s*FOR|\s*IN|\s*TO)?)\s*)?
    (?:
        (?:ZOOM\s*OR\s*)?OPEN\s*COURT
        |
        OPEN/ZOOM\s*COURT
        |
        ZOOM\s*(?:COURT|HEARING|MEETING)
        |
        COURT
        |
        TO\s*VIRTUAL\s*COURTROOM
    ))?
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY'?S\s*COURT\s*SETTING|TO\s*COME))?
    |
    # Federal detention facility patterns
    (?:IS|WAS)\s*IN\s*(?:THE\s*)?(?:SHERIFF\s*(?:IN\s*)?)?(?:DOC/\s*)?
    (?:{louisiana_parishes_pattern}|ANOTHER)
    (?:\s*PARISH)?(?:\s*SHERIFF\'?S?|\s*JAIL)?\s*
    (?:
        (?:FEDERAL\s*)?CUSTODY\s*(?:OF|IN)\s*(?:THE\s*)?(?:FEDERAL\s*)?
        (?:
            DETENTION\s*CENTER
            |
            BUREAU\s*OF\s*PRISONS
            |
            BOP|FDC|MCC|FCI|USP
            |
            MARSHALS?\s*SERVICE
            |
            ICE|IMMIGRATION
        )?
    )?\s*
    AND\s*(?:IS\s*|WAS\s*)?
    (?:
        NOT\s*(?:BROUGHT|TRANSPORTED|TRANSFERRED)
        |
        (?:REFUSED\s*|UNABLE\s*|FAILED\s*)
        (?:
            TO\s*(?:BE\s*(?:BROUGHT|TRANSPORTED)|APPEAR|COME|BROUGHT|LEAVE)
            |
            TRANSPORTED
        )
        |
        (?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE|BROUGHT(?:\s*OVER)?|TRANSPORTED(?:\s*OVER)?)|UNAVAILABLE|ABSENT)
        |
        DID\s*NOT\s*(?:APPEAR|COME|MAKE\s*AN\s*APPEARANCE)
        |
        NOT\s*AVAILABLE\s*THIS\s*DAY
    )
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?    
    (?:\s*
    (?:(?:TO|INTO|IN\s*TO|UP(?:\s*FOR|\s*IN|\s*TO)?)\s*)?
    (?:
        (?:ZOOM\s*OR\s*)?OPEN\s*COURT
        |
        OPEN/ZOOM\s*COURT
        |
        ZOOM\s*(?:COURT|HEARING|MEETING)
        |
        COURT
        |
        TO\s*VIRTUAL\s*COURTROOM
    ))?
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    |
    # General custody without court appearance
    IN\s*(?:THE\s*)?CUSTODY\s*AND\s*
    (?:IS\s*|WAS\s*)?
    (?:
        NOT\s*(?:BROUGHT|TRANSPORTED|TRANSFERRED)
        |
        (?:REFUSED\s*|UNABLE\s*|FAILED\s*)
        (?:
            TO\s*(?:BE\s*(?:BROUGHT|TRANSPORTED)|APPEAR|COME|BROUGHT|LEAVE)
            |
            TRANSPORTED
        )
        |
        (?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE|BROUGHT(?:\s*OVER)?|TRANSPORTED(?:\s*OVER)?)|UNAVAILABLE|ABSENT)
        |
        DID\s*NOT\s*(?:APPEAR|COME|MAKE\s*AN\s*APPEARANCE)
        |
        NOT\s*AVAILABLE\s*THIS\s*DAY
    )
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    |
    # Defendant did not appear
    DEFENDANT(?:\s*.\s*)?\s*DID\s*NOT\s*
    (?:APPEAR|COME(?:\s*TO)?|ATTEND|MAKE\s*AN\s*APPEARANCE)\s*
    (?:
        (?:ZOOM\s*OR\s*)?OPEN\s*COURT
        |
        OPEN/ZOOM\s*COURT
        |
        ZOOM\s*(?:COURT|HEARING|MEETING)
        |
        COURT
        |
        VIRTUAL\s*COURTROOM
    )?
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    |
    # Counsel appeared without defendant
    APPEARED\s*
    (?:
        WITHOUT\s*DEFENDANT
        |
        ON\s*BEHALF\s*OF\s*DEFENDANT
        |
        IN\s*PLACE\s*OF\s*DEFENDANT
        |
        IN\s*LIEU\s*OF\s*DEFENDANT
    )
    |
    # Special status conditions and not transported/did not appear
    (?:THE\s*)?DEFENDANT\s*(?:IS\s*|WAS\s*)?
    (?:STILL\s*|REMAINS\s*|CURRENTLY\s*)?
    (?:
        (?:IN|ON|IS|WAS)\s*(?:THE\s*|THIS\s*)?(?:DEXTOX|DETOX|HOSPITAL|JAIL|MEDICAL)
        |
        (?:IN|ON|IS|WAS)\s*(?:SUICIDE\s*WATCH|LOCKDOWN|ISOLATION)
        |
        (?:IN|ON|IS|WAS)\s*(?:THE\s*|THIS\s*)?(?:(?:MAX\.?\s*SECURITY|MEDICAL)\s*TIER)
        |
        A\s*SECURITY\s*ISSUE
        |
        (?:DETOXIFYING|DETOXYING)
        |
        (?:BEING\s*)?COMBATIVE
    )\s*
    (?:(?:@|AT)\s*(?:{all_louisiana_jails_pattern})\s*)?
    (?:
        AND\s*(?:IS\s*|WAS\s*)?
        (?:
            NOT\s*(?:BROUGHT|TRANSPORTED|TRANSFERRED)
            |
            (?:REFUSED\s*|UNABLE\s*|FAILED\s*)
            (?:
                TO\s*(?:BE\s*(?:BROUGHT|TRANSPORTED)|APPEAR|COME|BROUGHT|LEAVE)
                |
                TRANSPORTED
            )
            |
            (?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE|BROUGHT(?:\s*OVER)?|TRANSPORTED(?:\s*OVER)?)|UNAVAILABLE|ABSENT)
            |
            DID\s*NOT\s*(?:APPEAR|COME|MAKE\s*AN\s*APPEARANCE)
        )
        (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    )
    (?:\s*
        (?:(?:TO|INTO|IN\s*TO|UP(?:\s*FOR|\s*IN|\s*TO)?)\s*)?
        (?:
            (?:ZOOM\s*OR\s*)?OPEN\s*COURT
            |
            OPEN/ZOOM\s*COURT
            |
            ZOOM\s*(?:COURT|HEARING|MEETING)
            |
            COURT
            |
            TO\s*VIRTUAL\s*COURTROOM
        )
    )?
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'S\s*COURT\s*SETTING|TO\s*COME))?
    |
    ^NOT\s*(?:BROUGHT|TRANSPORTED)(?:\s*TO\s*(?:COURT|THIS\s*COURT\s*SETTING|TODAY\'S\s*COURT\s*SETTING|TODAY|VIRTUAL\s*COURTROOM|VIRTUAL\s*HEARING|VIRTUAL\s*MEETING|ZOOM\s*COURT|OPEN/ZOOM\s*COURT|OPEN\s*COURT|ZOOM\s*OR\s*OPEN\s*COURT))?
)
"""

## Compiler
def compile_custody_patterns(all_louisiana_jails_combined=all_louisiana_jails_combined, louisiana_parishes=louisiana_parishes):
    # Create pattern substitutions
    all_louisiana_jails_pattern = '|'.join(all_louisiana_jails_combined) if all_louisiana_jails_combined else r'[A-Z0-9\s\W]+'
    louisiana_parishes_pattern = '|'.join(louisiana_parishes) if louisiana_parishes else r'[A-Z]+'
    # Format the patterns with the jail and parish lists
    custody_pattern = in_custody_pattern.format(
        all_louisiana_jails_pattern=all_louisiana_jails_pattern,
        louisiana_parishes_pattern=louisiana_parishes_pattern
    )
    not_court_pattern = not_in_court_pattern.format(
        all_louisiana_jails_pattern=all_louisiana_jails_pattern,
        louisiana_parishes_pattern=louisiana_parishes_pattern
    )
    try:
        # Compile the patterns for better performance
        compiled_custody = re.compile(custody_pattern, flags=re.VERBOSE | re.IGNORECASE)
        compiled_not_court = re.compile(not_court_pattern, flags=re.VERBOSE | re.IGNORECASE)
        return compiled_custody, compiled_not_court
    except re.error as e:
        return None, None

compiled_custody_pattern, compiled_not_court_pattern = compile_custody_patterns(all_louisiana_jails_combined, louisiana_parishes)
## Extractor

def extract_defendant_in_custody_and_not_in_court(text):
    if text is None or not isinstance(text, str) or text.strip() == "":
        return "NONE"
    if compiled_custody_pattern is None or compiled_not_court_pattern is None:
        return "NONE"
    try:
        # Check if text matches both patterns using compiled regex
        match1 = compiled_custody_pattern.search(text)
        match2 = compiled_not_court_pattern.search(text)
        # Return classification based on pattern matches
        if match1 and match2:
            return "DEFENDANT_IN_CUSTODY_AND_NOT_IN_COURT"
        else:
            return "NONE"
    except Exception as e:
        return "NONE"

# SECTION State Not Present

## Pattern
state_not_present_pattern = r"""
(?:
    # ADA/AAG with absence terms
    (?:\b(?:A\.?)?D\.?A\.?\b\s*|\b(?:A\.?)?A\.?G\.?\b\s*)
    (?:(?:\'?S?\s*(?:OFFICE)?)?\s*)?
    (?:[\,\:]?\s*)?
    (?:[\w-]+(?:\.|\,|\')?\s+){0,4}\s*
    (?:[\u002D\u058A\u05BE\u2010\u2011\u2012\u2013\u2014\u2015\u2E3A\u2E3B\uFE58\uFE63\uFF0D]\s*)?
    (?:(?:WAS\s*|IS\s*)?(?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE)|UNAVAILABLE|ABSENT)|DID\s*NOT\s*(?:APPEAR|MAKE\s*AN\s*APPEARANCE))
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|STATE|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'s\s*COURT\s*SETTING|TO\s*COME))?
    (?:\s*(?:ON\s*BEHALF\s*OF\s*THE\s*STATE(?:\s*OF\s*LOUISIANA)?|TO\s*REPRESENT\s*(?:THE\s*(?:INTE?REST\s*OF\s*)?)?THE\s*STATE(?:\s*OF\s*LOUISIANA)?))?
    |
    # ADA/AAG with timing then absence
    (?:\b(?:A\.?)?D\.?A\.?\b\s*|\b(?:A\.?)?A\.?G\.?\b\s*)
    (?:(?:\'?S?\s*(?:OFFICE)?)?\s*)?
    (?:[\,\:]?\s*)?
    (?:[\w-]+(?:\.|\,|\')?\s+){0,4}\s*
    (?:[\u002D\u058A\u05BE\u2010\u2011\u2012\u2013\u2014\u2015\u2E3A\u2E3B\uFE58\uFE63\uFF0D]\s*)?
    (?:(?:FOR\s*|AT\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|STATE|SETTING|DAY)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'s\s*COURT\s*SETTING|TO\s*COME)?\s*
    (?:(?:WAS\s*|IS\s*)?(?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE)|UNAVAILABLE|ABSENT)|DID\s*NOT\s*(?:APPEAR|MAKE\s*AN\s*APPEARANCE))
    (?:\s*(?:ON\s*BEHALF\s*OF\s*THE\s*STATE(?:\s*OF\s*LOUISIANA)?|TO\s*REPRESENT\s*(?:THE\s*(?:INTE?REST\s*OF\s*)?)?THE\s*STATE(?:\s*OF\s*LOUISIANA)?))?
    |
    # ADA/AAG with representation then absence
    (?:\b(?:A\.?)?D\.?A\.?\b\s*|\b(?:A\.?)?A\.?G\.?\b\s*)
    (?:(?:\'?S?\s*(?:OFFICE)?)?\s*)?
    (?:[\,\:]?\s*)?
    (?:[\w-]+(?:\.|\,|\')?\s+){0,4}\s*
    (?:[\u002D\u058A\u05BE\u2010\u2011\u2012\u2013\u2014\u2015\u2E3A\u2E3B\uFE58\uFE63\uFF0D]\s*)?
    (?:(?:FOR\s*|AT\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|STATE|SETTING|DAY)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'s\s*COURT\s*SETTING|TO\s*COME)?\s*
    (?:\s*(?:(?:IS\s*|WAS\s*)(?:HANDLING\s*THIS\s*MATTER\s*(?:FOR\s*|ON\s*BEHALF\s*OF\s*)THE\s*STATE(?:\s*OF\s*LOUISIANA)?)|REPRESENTING\s*THE\s*STATE(?:\s*OF\s*LOUISIANA)?|REPRESENTS?\s*THE\s*STATE(?:\s*OF\s*LOUISIANA)?|ON\s*BEHALF\s*OF\s*THE\s*STATE(?:\s*OF\s*LOUISIANA)?|TO\s*REPRESENT\s*(?:THE\s*(?:INTE?REST\s*OF\s*)?)?THE\s*STATE(?:\s*OF\s*LOUISIANA)?))?
    (?:\s*(?:IN\s*THIS\s*MATTER|FOR\s*THIS\s*(?:MATTER|CASE|PROCEE?DING)))?
    \s*(?:AND\s*|,?\s*BUT\s*)?
    (?:(?:WAS\s*|IS\s*)?(?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE)|UNAVAILABLE|ABSENT)|DID\s*NOT\s*(?:APPEAR|MAKE\s*AN\s*APPEARANCE))
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|STATE|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'s\s*COURT\s*SETTING|TO\s*COME))?
    |
    # State of Louisiana not present
    STATE(?:\s+(?:OF|O|AF)\s+LOUI?SIANA)?\s+
    (?:WAS\s*|IS\s*)?
    (?:SERVED\s*)?
    (?:\,?\s*BUT\s*)?
    (?:(?:WAS\s*|IS\s*)?(?:NOT\s*(?:PRESENT|AVAILABLE)|UNAVAILABLE|ABSENT)|DID\s+NOT\s+(?:HAVE\s+A\s+REPRESENTATIVE\s+PRESENT|APPEAR|PRESENT))
    (?:\s*(?:FOR\s*|ON\s*|IN\s*|AS\s*|AT\s*|TODAY)(?:THE\s*|THIS\s*|ANY\s*|A\s*|AN\s*|THAT\s*)?(?:HEARING|BOND\s*SETTING|MATTER(?:S\s*HEARD\s*IN\s*MAGISTRATE\s*COURT)?|COURT|PROCEE?DING)?)?
    |
    # District Attorney/Attorney General office absence
    (?:(?:(?:ASSISTANT|ASST\.?)\s*)?DIST(?:RICT|\.)?\s*(?:ATTORNEY|ATTY\.?)(?:\'?S?\s*(?:OFFICE)?)?\s*|(?:(?:ASSISTANT|ASST\.?)\s*)?(?:ATTORNEY|ATTY\.?)\s*(?:GENERAL|GEN\.?)(?:\'?S\s*OFFICE)?\s*)
    (?:[\,\:]?\s*)?
    (?:[\w-]+(?:\.|\,|\')?\s+){0,4}
    (?:[\u002D\u058A\u05BE\u2010\u2011\u2012\u2013\u2014\u2015\u2E3A\u2E3B\uFE58\uFE63\uFF0D]\s*)?
    (?:(?:WAS\s*|IS\s*)?(?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE)|UNAVAILABLE|ABSENT)|DID\s*NOT\s*(?:APPEAR|MAKE\s*AN\s*APPEARANCE))
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|STATE|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'s\s*COURT\s*SETTING|TO\s*COME))?
    (?:\s*(?:ON\s*BEHALF\s*OF\s*THE\s*STATE(?:\s*OF\s*LOUISIANA)?|TO\s*REPRESENT\s*(?:THE\s*(?:INTE?REST\s*OF\s*)?)?THE\s*STATE(?:\s*OF\s*LOUISIANA)?))?
    |
    # District Attorney/Attorney General office with timing then absence
    (?:(?:(?:ASSISTANT|ASST\.?)\s*)?DIST(?:RICT|\.)?\s*(?:ATTORNEY|ATTY\.?)(?:\'?S\s*(?:OFFICE)?)?\s*|(?:(?:ASSISTANT|ASST\.?)\s*)?(?:ATTORNEY|ATTY\.?)\s*(?:GENERAL|GEN\.?)(?:\'?S?\s*OFFICE)?\s*)
    (?:[\,\:]?\s*)?
    (?:[\w-]+(?:\.|\,|\')?\s+){0,4}\s*
    (?:[\u002D\u058A\u05BE\u2010\u2011\u2012\u2013\u2014\u2015\u2E3A\u2E3B\uFE58\uFE63\uFF0D]\s*)?
    (?:(?:FOR\s*|AT\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|STATE|SETTING|DAY)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'s\s*COURT\s*SETTING|TO\s*COME)?\s*
    (?:(?:WAS\s*|IS\s*)?(?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE)|UNAVAILABLE|ABSENT)|DID\s*NOT\s*(?:APPEAR|MAKE\s*AN\s*APPEARANCE))
    (?:\s*(?:ON\s*BEHALF\s*OF\s*THE\s*STATE(?:\s*OF\s*LOUISIANA)?|TO\s*REPRESENT\s*(?:THE\s*(?:INTE?REST\s*OF\s*)?)?THE\s*STATE(?:\s*OF\s*LOUISIANA)?))?
    |
    # District Attorney/Attorney General office with representation then absence
    (?:(?:(?:ASSISTANT|ASST\.?)\s*)?DIST(?:RICT|\.)?\s*(?:ATTORNEY|ATTY\.?)(?:\'?S?\s*(?:OFFICE)?)?\s*|(?:(?:ASSISTANT|ASST\.?)\s*)?(?:ATTORNEY|ATTY\.?)\s*(?:GENERAL|GEN\.?)(?:\'?S?\s*OFFICE)?\s*)
    (?:[\,\:]?\s*)?
    (?:[\w-]+(?:\.|\,|\')?\s+){0,4}\s*
    (?:[\u002D\u058A\u05BE\u2010\u2011\u2012\u2013\u2014\u2015\u2E3A\u2E3B\uFE58\uFE63\uFF0D]\s*)?
    (?:(?:FOR\s*|AT\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|STATE|SETTING|DAY)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'s\s*COURT\s*SETTING|TO\s*COME)?\s*
    (?:\s*(?:(?:IS\s*|WAS\s*)(?:HANDLING\s*THIS\s*MATTER\s*(?:FOR\s*|ON\s*BEHALF\s*OF\s*)THE\s*STATE(?:\s*OF\s*LOUISIANA)?)|REPRESENTING\s*THE\s*STATE(?:\s*OF\s*LOUISIANA)?|REPRESENTS?\s*THE\s*STATE(?:\s*OF\s*LOUISIANA)?|ON\s*BEHALF\s*OF\s*THE\s*STATE(?:\s*OF\s*LOUISIANA)?|TO\s*REPRESENT\s*(?:THE\s*(?:INTE?REST\s*OF\s*)?)?THE\s*STATE(?:\s*OF\s*LOUISIANA)?))?
    (?:\s*(?:IN\s*THIS\s*MATTER|FOR\s*THIS\s*(?:MATTER|CASE|PROCEE?DING)))?
    \s*(?:AND\s*|,?\s*BUT\s*)?
    (?:(?:WAS\s*|IS\s*)?(?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE)|UNAVAILABLE|ABSENT)|DID\s*NOT\s*(?:APPEAR|MAKE\s*AN\s*APPEARANCE))
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|STATE|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'s\s*COURT\s*SETTING|TO\s*COME))?
    |
    # Representing state and absent
    (?:REPRESENT(?:S|ING)?\s*(?:THE\s*(?:INTE?REST\s*OF\s*)?)?THE\s*STATE(?:\s*OF\s*LOUISIANA)?)
    (?:\s*(?:ON\s*|IN\s*)(?:THE\s*|THIS\s*|A\s*|AN\s*|THAT\s*)?(?:HEARING|MATTER|COURT|PROCEE?DING)?)?
    (?:\s*AND\s*|\s*\,\s*)
    (?:(?:WAS\s*|IS\s*)?(?:NOT\s*(?:PRESENT|AVAILABLE)|UNAVAILABLE|ABSENT)|DID\s*NOT\s*(?:APPEAR|MAKE\s*AN\s*APPEARANCE))
    (?:\s*(?:(?:FOR\s*|AT\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|STATE|SETTING|DAY)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TO\s*COME))?
    |
    # For the state and absent
    (?:FOR\s*THE\s*STATE(?:\s*OF\s*LOUISIANA)?|REPRESENTING\s*THE\s*STATE(?:\s*OF\s*LOUISIANA)?|ON\s*BEHALF\s*OF\s*THE\s*STATE(?:\s*OF\s*LOUISIANA)?)\s*
    (?:[\,\:]?\s*)?
    (?:(?:WAS\s*|IS\s*)?(?:NOT\s*(?:BEING\s*)?(?:PRESENT|AVAILABLE)|UNAVAILABLE|ABSENT)|DID\s*NOT\s*(?:APPEAR|MAKE\s*AN\s*APPEARANCE))
    (?:\s*(?:(?:FOR\s*|AT\s*|ON\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|STATE|SETTING|DAY|MATTER)|DAY|IN\s*COURT(?:\s*TODAY)?|TODAY|TODAY\'s\s*COURT\s*SETTING|TO\s*COME))?
    (?:\s*(?:ON\s*BEHALF\s*OF\s*THE\s*STATE(?:\s*OF\s*LOUISIANA)?|TO\s*REPRESENT\s*(?:THE\s*(?:INTE?REST\s*OF\s*)?)?THE\s*STATE(?:\s*OF\s*LOUISIANA)?))?
    |
    # No one present to represent
    NO\s*ONE\s*(?:PRESENT|APPEAR(?:ED|S|ING)?|AVAILABLE)\s*
    (?:TO\s*REPRESENT\s*|ON\s*BEHALF\s*OF\s*)?(?:THE\s*)?
    (?:(?:(?:(?:ASSISTANT|ASST\.?)\s+)?DIST(?:RICT|\.)?\s*(?:ATTORNEY|ATTY\.?)(?:\'?S?\s*(?:OFFICE)?)?|(?:(?:ASSISTANT|ASST\.?)\s+)?(?:ATTORNEY|ATTY\.?)\s+(?:GENERAL|GEN\.?)(?:\'?S?\s*(?:OFFICE)?)?|STATE(?:\s+OF\s+LOUISIANA)?|\b(?:A\.?)?D\.?A\.?\b\s*|\b(?:A\.?)?A\.?G\.?\b\s*))
    |
    # Office did not send attorney
    (?:AS\s*)?THEIR\s*OFFICE\s*DID\s*NOT\s*SEND\s*(?:AN\s*|A\s*)?(?:ASSISTANT\s*|ASST\.?\s*)?(?:DIST(?:RICT|\.)?\s*(?:ATTORNEY|ATTY\.?)|(?:ATTORNEY|ATTY\.?)\s*(?:GENERAL|GEN\.?))\s*
    TO\s*REPRESENT\s*(?:THE\s*INTE?REST\s*OF\s*)?THE\s*STATE(?:\s*OF\s*LOUISIANA)?
    |
    # DA/AG ordering assistants not to appear
    (?:(?:A\.?D\.?A\.?\'?S?|A\.?A\.?G\.?\'?S?|(?:ASSISTANT|ASST\.?)\s*DIST(?:RICT|\.)?\s*(?:ATTORNEY|ATTY\.?)\'?S?|(?:ASSISTANT|ASST\.?)\s*(?:ATTORNEY|ATTY\.?)\s*(?:GENERAL|GEN\.?)\'?S?)\s*
    (?:WERE\s*|WAS\s*|ARE\s*|IS\s*)?
    (?:NOT\s*(?:PRESENT|AVAILABLE)|UNAVAILABLE|ABSENT|DID\s*NOT\s*(?:APPEAR|MAKE\s*AN\s*APPEARANCE))
    (?:\s*(?:IN\s*COURT|(?:FOR\s*|AT\s*)?(?:THIS\s*|THE\s*)?(?:PROCEE?DING|HEARING|STATE|SETTING|DAY)|DAY|TODAY))?
    \s*DUE\s*TO\s*
    (?:THE\s*)?(?:DIST(?:RICT|\.)?\s*(?:ATTORNEY|ATTY\.?)|(?:ATTORNEY|ATTY\.?)\s*(?:GENERAL|GEN\.?)|D\.?A\.?|A\.?G\.?)
    \s*ORDERING\s*(?:THE\s*|THEIR\s*)?(?:ASSISTANTS?|ASSTS?\.?)\s*
    (?:TO\s*)?NOT\s*(?:(?:TO\s*)?BE\s*PRESENT|(?:TO\s*)?APPEAR|(?:TO\s*)?MAKE\s*AN\s*APPEARANCE)
    (?:\s*IN\s*COURT)?
    (?:\s*\.)?)
)
"""
state_not_present_compiled = regex.compile(state_not_present_pattern, flags = regex.VERBOSE | regex.IGNORECASE)

## Extractor
def extract_the_state_was_not_present(text):
    if text is None or not isinstance(text, str) or text.strip() == "":
        return "NONE"
    try:
        match = state_not_present_compiled.search(text)
        if match:
            return "THE_STATE_WAS_NOT_PRESENT"
        else:
            return "NONE"
    except Exception as e:
        return "NONE"
    
# SECTION Pre-Trial Conference Occured

## Pattern

pre_trial_conference_pattern = r"""
(?:
    # Appeared before court with counsel for pre-trial conference
    APPEARED\s*
    (?:BEFORE\s*THE\s*COURT\s*)?\s*
    WITH\s*COUNSEL\s*
    (?:
        (?:\,)?\s*
        (?:\w*(?:\.|\,|\')?\s*){1,3}
        (?:\,)?
        (?:
            \s*VIA\s*ZOOM
            |
            \s*STANDING\s*IN
            (?:\s*FOR\s*(?:\w*(?:\.|\,|\')?\s*){1,5})?
            (?:\s*\,\s*)?
        )?
    )\s*
    (?:FOR\s*PRE(?:\s*.\s*|\s*)?TRIAL\s*CONFERENCE)
    |
    # Appeared before court for pre-trial conference with counsel
    APPEARED\s*
    (?:BEFORE\s*THE\s*COURT\s*)?
    FOR\s*PRE(?:\s*.\s*|\s*)?TRIAL\s*CONFERENCE\s*
    WITH\s*COUNSEL
    |
    # Pre-trial conference was held
    PRE(?:\s*.\s*|\s*)?TRIAL\s*CONFERENCE\s*
    (?:WAS\s*)?HELD
    |
    # Appeared for pre-trial conference (duplicate pattern variant)
    APPEARED\s*
    (?:BEFORE\s*THE\s*COURT\s*)?
    FOR\s*PRE(?:\s*.\s*|\s*)?TRIAL\s*
    (?:CONFERENCE\s*)?
    (?:FOR\s*)?
    PRE(?:\s*.\s*|\s*)?TRIAL\s*CONFERENCE\s*
    WITH\s*COUNSEL
    |
    # Pre-trial conference via zoom with counsel
    PRE(?:\s*.\s*|\s*)?TRIAL\s*CONFERENCE\s*
    (?:VIA\s*ZOOM\s*)?
    (?:\,\s*)?
    WITH\s*COUNSEL
    |
    # Appeared via zoom with counsel
    APPEARED\s*
    (?:BEFORE\s*THE\s*COURT\s*)?
    (?:VIA\s*ZOOM\s*)?
    (?:\,\s*)?
    WITH\s*COUNSEL\s*
    (?:
        (?:\,)?\s*
        (?:\w*(?:\.|\,|\')?\s*){1,3}
        (?:\,)?
        (?:
            \s*VIA\s*ZOOM
            |
            \s*STANDING\s*IN
            (?:\s*FOR\s*(?:\w*(?:\.|\,|\')?\s*){1,5})?
            (?:\s*\,\s*)?
        )?
    )\s*
    (?:FOR\s*PRE(?:\s*.\s*|\s*)?TRIAL\s*CONFERENCE)
    |
    # Defense counsel appeared for pre-trial conference
    DEFENSE\s*COUNSEL\s*
    (?:\w*(?:\.|\,|\')?\s*){1,3}\s*
    (?:\,)?\s*
    APPEARED\s*
    (?:BEFORE\s*THE\s*COURT\s*)?
    (?:VIA\s*ZOOM\s*)?
    (?:\,\s*)?
    (?:
        \s*WITHOUT\s*DEFENDANT\s*
        |
        \s*ON\s*BEHALF\s*OF\s*DEFENDANT\s*
    )?
    (?:\,\s*)?
    (?:\w*(?:\.|\,|\')?\s*){1,3}
    (?:\,)?\s*
    FOR\s*PRE(?:\s*.\s*|\s*)?TRIAL\s*CONFERENCE
)
"""
compiled_pretrial_pattern = re.compile(pre_trial_conference_pattern, flags = re.VERBOSE | re.IGNORECASE)

## Extractor
def extract_pre_trial_conference(text):
    if text is None or not isinstance(text, str) or text.strip() == "":
        return "NONE"
    try:
        # Check if text matches the pre-trial conference pattern
        match = compiled_pretrial_pattern.search(text)
        if match:
            return "PRE_TRIAL_CONFERENCE_HELD"
        else:
            return "NONE"
    except Exception as e:
        return "NONE"

# SECTION Continuance Occured

## Patterns

### Result Based Patterns
result_patterns = [
    r'RESULT[-\s]*CONTINUED(?:\s+WITHOUT\s+DATE)?',
    r'RESULT[-\s]*CONTINUED\s+BY\s+THE\s+COURT',
    r'RESULT[-\s]*CONTINUED\s+FOR\s+THE\s+(?:DEFENSE|STATE)',
    r'RESULT[-\s]*CONTINUED\s+FOR\s+\d+/\d+/\d+\s+AT\s+\d+:\d+\s+[AP]M',
    r'RESULT[-\s]*CONTINUED\s+BY\s+JOINT\s+MOTION',
    r'RESULT[-\s]*(?:STATUS\s+HEARING|RULE\s+TO\s+SHOW\s+CAUSE|PRELIMINARY\s+HEARING)\s+CONTINUED',
    r'RESULT[-\s]*RESET',
]

### Reset Patterns
reset_patterns = [
    # Court reset patterns
    r'COURT\s+RESET(?:\s+THE\s+MATTER|\s+THIS\s+MATTER)?',
    r'COURT\s+RESET\s+THIS\s+(?:PRE-TRIAL\s+CONFERENCE|ARRAIGNMENT)',
    r'COURT\s+RESET\s+(?:THE\s+)?(?:PRE-TRIAL\s+CONFERENCE|ARRAIGNMENT)',
    r'RESET\s+BY\s+(?:COURT|STATE|DEFENSE)',
    r'RESET,\s+(?:DEFENDANT\s+NOT\s+SERVED|TRIAL\s+IN\s+PROGRESS)',
    r'RESET\s+THIS\s+HEARING\s+ON\s+MOTIONS',
    r'>THE\s+COURT\s+RESET\s+THIS',

    # Matter reset patterns
    r'(?:THIS\s+)?MATTER\s+(?:IS|WAS|HAS\s+BEEN)?\s*RESET',
    r'RESET\s+(?:THE\s+|THIS\s+)?MATTER',
    r'THE\s+COURT\s+ORDERED\s+THIS\s+MATTER\s+RESET',
    r'RESET\s+AS\s+PER\s+THE\s+COURT',
    r'THE\s+RESET\s+OF\s+A\s+NEW\s+DATE\s+IN\s+THIS\s+MATTER',

    # Hearing type reset patterns
    r'(?:ARRAIGNMENT|HEARING|TRIAL|CASE)\s+(?:IS\s+|WAS\s+|HAS\s+BEEN\s+)?RESET',
    r'RESET\s+(?:ARRAIGNMENT|HEARING|TRIAL)',
    r'(?:STATUS\s+HEARING|PRELIMINARY\s+HEARING|BOND\s+(?:NOTIFICATION|SETTING|HEARING|REDUCTION))\s+RESET',
    r'RESET\s+(?:STATUS\s+HEARING|PRELIMINARY\s+HEARING|BOND\s+(?:NOTIFICATION|SETTING|HEARING|REDUCTION))',
    r'(?:DOMESTIC\s+STATUS(?:\s+HEARING)?|DOMESTIC\s+HEARING)\s+RESET',
    r'RESET\s+(?:DOMESTIC\s+STATUS(?:\s+HEARING)?|DOMESTIC\s+HEARING)',
    r'(?:RTSC|RULE\s+TO\s+SHOW\s+CAUSE)\s+RESET',
    r'RESET\s+(?:THE\s+)?(?:RTSC|RULE\s+TO\s+SHOW\s+CAUSE)',
    r'(?:PRE-TRIAL|PRETRIAL)\s+CONFERENCE\s+(?:IS\s+|WAS\s+|HAS\s+BEEN\s+)?RESET',
    r'RESET\s+(?:PRE-TRIAL|PRETRIAL)\s+CONFERENCE',
    r'(?:PROBATION\s+STATUS(?:\s+HEARING)?)\s+(?:IS\s+|WAS\s+|HAS\s+BEEN\s+)?RESET',
    r'RESET\s+PROBATION\s+STATUS(?:\s+HEARING)?',
    r'(?:DRUG\s+COURT\s+STATUS\s+HEARING)\s+(?:IS\s+|WAS\s+|HAS\s+BEEN\s+)?RESET',
    r'RESET\s+DRUG\s+COURT\s+STATUS\s+HEARING',
    r'HEARING\s+ON\s+MOTIONS\s+(?:IS\s+|WAS\s+|HAS\s+BEEN\s+)?RESET',
    r'RESET\s+HEARING\s+ON\s+MOTIONS',

    # Time-based reset patterns
    r'RESET\s+(?:FOR\s+)?(?:1|ONE)\s+(?:DAY|WEEK)',
    r'RESET\s+(?:FOR\s+)?(?:2|TWO)\s+WEEKS?',
    r'RESET\s+TO\s+',
    r'DAY\s+RESET',
    r'RESET\s+DUE\s+TO',
    r'THIS\s+IS\s+A\s+RESET',

    # Special reset patterns
    r'HEARING\s+RESET\s+FOR',
    r'WAS\s+RESET',
    r'COURT\s+WAS\s+CLOSED\s+DUE\s+TO\s+WEATHER\s+COURT\s+RESET',
    r'RESETS\s+THE\s+PREVIOUSLY\s+(?:SCHEDULED|SET)\s+HEARING',
    r'RESET\s+ON\s+JOINT\s+MOTION',
    r'(?:STATE|HURRICANE\s+IDA|CORONAVIRUS)\s+RESET',
    r'COURT,\s*ON\s+ITS\s+OWN\s+MOTION,\s*RESETS',

    # Simple reset patterns
    r'^\s*RESET\s*(?:-|$)',
    r'RESET[-\s]*',
]

### Rolled Patterns
rolled_patterns = [
    r'COURT\s+ROLLED',
    r'(?:CASE|MATTER)\s+(?:IS\s+|WAS\s+)?ROLLED',
    r'(?:IS|WAS)\s+ROLLED\s+OVER',
    r'MATTER\s+ROLLED\s+OVER',
    r'WILL\s+BE\s+ROLLED\s+OVER',
    r'ARRAIGNMENT\s+ROLLED\s+OVER',
    r'COURT\s+ROLLED\s+THIS\s+MATTER',
    r'WILL\s+ROLL\s+THIS\s+MATTER',
    r'ROLLING\s+THIS\s+MATTER',
    r'ROLLED\s+OVER\s+ONE',
    r'THE\s+COURT\s+ROLLED',
    r'ROLLED\s+(?:THE\s+)?MATTER',
    r'ROLLED\s+THIS\s+MATTER',
    r'ROLLED\s+UNTIL',

    # Specific day/time rolled patterns
    r'ROLLED?\s+TO\s+(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)',
    r'ROLLED?\s+(?:TO\s+|UNTIL\s+)?TOMORROW',
    r'ROLLED?\s+MATTER\s+TO\s+TOMORROW',
    r'ROLLED\s+TO\s+NEXT\s+(?:DAY|WEEK|MONDAY)',
    r'WAS\s+ROLLED\s+TO\s+',
    r'ROLL\s+(?:ONE\s+)?(?:DAY|WEEK)',
    r'ROLL\s+1-DAY',

    # Special rolled patterns
    r'ROLLED\s+THE\s+RTSC',
    r'HEARING\s+ROLLED',
    r'ROLL\s+THE\s+DEFENDANT',
    r'MATTER\s+(?:HAS\s+BEEN\s+)?ROLLED',
]

### Rescheduled Patterns
rescheduled_patterns = [
    r'THIS\s+MATTER\s+(?:IS|WAS|HAS\s+BEEN)?\s*RESCHEDULED',
    r'THIS\s+MATTER\s+WAS\s+RESCHEDULED\s+DUE\s+TO\s+THE\s+COURT\'S\s+CLOSURE\s+DURING\s+HURRICANE\s+IDA',
    r'THE\s+COURT\s+HAS\s+RESCHEDULED\s+THIS\s+MATTER\s+DUE\s+TO\s+INCLEMENT\s+WEATHER\s+CAUSING\s+COURT\s+CLOSURE',
    r'RESCHEDULE',
]

### Continued Patterns
continued_patterns = [
    # Arrow-prefix patterns
    r'>\s*CONTINUED\s+ON\s+(?:JOINT|DEFENSE|STATE)\s+MOTION',
    r'>\s*CONTINUED\s+ON\s+MOTION\s+BY\s+(?:THE\s+)?(?:STATE|DEFENSE)',
    r'>\s*CONTINUED\s+(?:WITHOUT\s+DATE|FOR\s+(?:DEFENSE|STATE))',
    r'>\s*TRIAL\s+CONTINUED',

    # Court continued patterns
    r'COURT\s+CONTINUED(?:\s+THE\s+MATTER)?',
    r'THE\s+COURT\s+CONTINUED\s+THIS\s+(?:MOTION|MATTER)(?:\s+WITHOUT\s+DATE)?',
    r'THE\s+COURT\s+(?:IS\s+)?CONTINUED(?:\s+WITHOUT\s+DATE)?',
    r'THE\s+COURT\s+CONTINUED\s+THE\s+MATTER\s+FOR\s+THE\s+(?:STATE|DEFENSE)',
    r'COURT,?\s*ON\s+ITS\s+OWN\s+MOTION,?\s*CONTINUES',
    r'COURT\s+CONTINUED\s+PH\.\s+ON\s+\d+/\d+/\d+\s+WITHOUT\s+DATE',

    # Matter continued patterns
    r'THIS\s+MATTER\s+(?:IS|WAS|HAS\s+BEEN)?\s*CONTINUED?(?:\s+WITHOUT\s+DATE)?',
    r'THIS\s+MATTER\s+(?:IS|WAS)\s+CONTINUED\s+(?:AT\s+THE\s+REQUEST\s+OF|BY)\s+(?:THE\s+)?(?:STATE|DEFENSE)',
    r'THIS\s+MATTER\s+(?:IS|WAS|HAS\s+BEEN)?\s*CONTINUED\s+(?:JOINTLY|THROUGH\s+A\s+JOINT\s+CONTINUANCE)',
    r'THIS\s+MATTER\s+BE\s+CONTINUED',

    # Other continued patterns
    r'PH\.?,?\s*CONTINUED\s+WITHOUT\s+DATE',
    r'AT\s+THE\s+REQUEST\s+OF\s+.+?\s+THE\s+COURT\s+CONTINUED\s+THIS\s+MATTER',
]

### Continuance Patterns (Explicit Use Of The Word)
continuance_patterns = [
    # Simple continuance statements
    r'THIS\s+(?:IS|WAS)\s+A?\s*(?:STATE|DEFENSE|JOINT|COURT)\s+CONTINUANCE',
    r'THIS\s+MATTER\s+(?:IS|WAS)\s+(?:A\s+)?(?:STATE|DEFENSE|JOINT)\s+CONTINUANCE',

    # Court granted patterns
    r'THE\s+COURT\s+GRANTED\s+(?:THE\s+)?(?:DEFENSE|STATE)\s+(?:A\s+)?CONTINUANCE',
    r'THE\s+COURT\s+GRANTED\s+(?:THE\s+)?CONTINUANCE',
    r'THE\s+COURT\s+GRANTED\s+(?:A\s+)?(?:DEFENSE\'?S?|STATE\'?S?|JOINT)\s+CONTINUANCE',
    r'THE\s+COURT\s+GRANTED\s+(?:THE\s+)?(?:DEFENSE\'?S?|STATE\'?S?)\s+MOTION\s+FOR\s+(?:A\s+)?CONTINUANCE',
    r'THE\s+COURT\s+GRANTED\s+(?:THE\s+)?MOTION\s+TO\s+CONTINUE(?:\s+WITHOUT\s+DATE)?',
    r'THE\s+COURT\s+GRANTED\s+THIS\s+CONTINUANCE',
    r'THE\s+COURT\s+GRANTED\s+(?:THE\s+)?(?:DEFENSE\'?S?|STATE\'?S?)\s+REQUEST\s+FOR\s+(?:A\s+)?(?:1\s+DAY\s+)?CONTINUANCE',
    r'THE\s+COURT\s+GRANTED\s+(?:THE\s+)?MOTION\s+FOR\s+(?:A\s+)?(?:DEFENSE|STATE)\s+CONTINUANCE',
    r'THE\s+COURT\s+GRANTED\s+(?:THE\s+)?JOINT\s+MOTION(?:\s+TO\s+CONTINUE)?',
    r'THE\s+STATE\s+REQUESTED\s+A\s+CONTINUANCE\s+WHICH\s+WAS\s+GRANTED\s+BY\s+THE\s+COURT',

    # Single line patterns
    r'^\s*(?:STATE|DEFENSE|JOINT|COURT)\s+CONT(?:INUANCE)?\.{0,3}\s*$',
    r'^\s*PRLIM\s+CONT\.{0,3}\s*$',

    # Granted patterns
    r'(?:DEFENSE|STATE|JOINT|1\s+DAY)\s+CONTINUANCE[-\s]*(?:REQUESTED\s+AND\s+)?GRANTED',
    r'(?:DEFENSE|STATE|JOINT)\s+MOTION\s+FOR\s+CONTINUANCE\s+WAS\s+GRANTED',
    r'CONTINUANCE[-\s]*GRANTED',
    r'GRANTED\s+(?:A\s+)?(?:1\s+DAY|DEFENSE|STATE|JOINT)?\s*CONTINUANCE',
    r'COURT\s+GRANT(?:ED)?\s+(?:A\s+)?(?:THE\s+)?(?:DEFENSE|STATE|JOINT|1\s+DAY)?\s*CONTINUANCE',
    r'COURT\s+GRANTED\s+(?:THE\s+)?(?:DEFENSE|STATE)\s+(?:A\s+)?CONTINUANCE(?:\s+FOR\s+ONE\s+WEEK)?',
    r'COURT\s+GRANTED\s+(?:THE\s+)?(?:DEFENSE\'?S?|STATE\'?S?|JOINT)\s+MOTION\s+FOR\s+(?:A\s+)?CONTINUANCE',
    r'COURT\s+GRANTED\s+(?:THE\s+)?MOTION\s+TO\s+CONTINUE(?:\s+RULE\s+TO\s+SHOW\s+CAUSE)?',
    r'MOTION\s+TO\s+CONTINUE\s+(?:PRLIM|PRELIM|PRELIMINARY\s+HEARING|PH\.?|P\.H\.)\s+GRANTED',
    r'JUDGE\s+GRANT(?:ED)?\s+MOTION\s+TO\s+CONTINUE',

    # Other continuance patterns
    r'CONTINUANCE\s+REQUESTED\s+BY\s+.+?\s+THE\s+COURT\s+GRANTED\s+THE\s+MOTION',
    r'THIS\s+MATTER\s+(?:IS|WAS)\s+RESET\s+BY\s+(?:A\s+)?(?:DEFENSE|STATE|JOINT)?\s*CONTINUANCE',
    r'THIS\s+MATTER\s+(?:IS|WAS|HAS\s+BEEN)?\s*CONTINUED\s+BY\s+(?:A\s+)?(?:DEFENSE|STATE|JOINT)\s+CONTINUANCE',
    r'>\s*CONTINUANCE\s+ON\s+(?:DEFENSE|STATE|JOINT)?\s*MOTION',
]

### Refix Patterns (For Hearings/Dates)
refix_patterns = [
    r'COURT\s+GRANTS\s+MOTION\s+TO\s+REFIX',
]

## Combining all of these groups of continuance patterns into one group of patterns and then a regular expression
all_continuance_patterns = (
    result_patterns +
    reset_patterns +
    rolled_patterns +
    rescheduled_patterns +
    continued_patterns +
    continuance_patterns +
    refix_patterns
)
CONTINUANCE_REGEX = '|'.join(f'({pattern})' for pattern in all_continuance_patterns)
CONTINUANCE_PATTERN = re.compile(CONTINUANCE_REGEX, re.IGNORECASE | re.MULTILINE)

## Patterns that should be excluded (not considered continuances) into a regular expression
exclusion_patterns = [
    # Bond/bail fixing (not a continuance)
    r'(?=.*(BOND|BAIL)).*\b\w*FIX\w*\b',
    # Filed/requested but not granted
    r'(?!.*GRANTED)(?=.*(FILED|REQUESTED)).*\b\w*RESET\w*\b',
    r'(?!.*GRANTED)(?=.*(FILED|REQUESTED)).*\b\w*CONTINUE\w*\b',
    # Previously rescheduled (already happened)
    r'PREVIOUSLY\s+RESCHEDULED',
    # Appeared for reset only (not an actual reset)
    r'APPEARED\s+FOR\s+RESET\s+ONLY',
    # Progress report reset (specific exclusion from document)
    r'PROGRESS\s+REPORT\s+RESET',
]
EXCLUSION_REGEX = '|'.join(f'({pattern})' for pattern in exclusion_patterns)
EXCLUSION_PATTERN = re.compile(EXCLUSION_REGEX, re.IGNORECASE | re.MULTILINE)

## Extractor
def determine_continuance(proceeding):
    def is_continuance(text):
        if pd.isna(text):
            return False
        # Check if it matches a delay pattern
        has_continuance = bool(CONTINUANCE_PATTERN.search(text))
        # Check if it matches an exclusion pattern
        has_exclusion = bool(EXCLUSION_PATTERN.search(text))
        # Return True only if it has a delay pattern and no exclusion
        return has_continuance and not has_exclusion
    if is_continuance(proceeding):
        return 'CONTINUANCE_OCCURRED'
    else:
        return 'NONE'
    


# SECTION Plea Status

## Pattern
plea_pattern = re.compile(r'(?:(?:ENTERS?|ENTERED|ENTERING)\s+(?:AMENDED\s+THE\s+DEFENDANT[\'"]S\s+PLEA\s+TO\s+A\s+PLEA\s+OF\s+OF\s+NOT\s+GUILTY\s+AND\s+NOT\s+GUILTY\s+BY\s+REASON\s+OF\s+INSANITY|(?:A\s+)?(?:AN\s+)?(?:AMENDED\s+)?(?:PLEA\s+(?:OF\s+)?(?:NOT\s+)?GUILTY|(?:NOT\s+)?GUILTY\s+PLEA)(?:\s+IN\s+ABSENTIA)?(?:\s+\[X\])?)|(?:PLEDS?|PLEADS?|PLED|PLEAD|PLEADING)\s+(?:NOT\s+)?GUILTY(?:\s+AS\s+CHARGED|\s+PURSUANT\s+TO|\s+UNDER\s+ART)?|GUILTY\s+PLEA(?:\s+ENTERED)?|A\s+PLEA\s+OF\s+(?:NOT\s+)?GUILTY|GUILTY\s+PURSUANT\s+TO|(?:ENTERS?|ENTERED|ENTERING)\s+(?:A\s+)?(?:PLEA\s+OF\s+)?(?:NO\s+CONTEST|NOLO\s+CONTENDERE)(?:\s+TO)?)')

## Extractor

def extract_plea_status(text):
    if not isinstance(text, str):
        return None
    m = plea_pattern.findall(text)
    results = []
    for match in m:
        # Convert to uppercase for consistent matching
        match_upper = match.upper()
        # split match_upper and rejoin it
        match_upper = ' '.join(match_upper.split())
        if 'NOT GUILTY' in match_upper:
            results.append('NOT_GUILTY')
        elif 'GUILTY' in match_upper:
            results.append('GUILTY')
        else:
            results.append('NOT APPLICABLE')
    return results

def extract_unique_plea_set(text: str) -> set[str]:
    raw = extract_plea_status(text) or []
    # drop N/A and map
    raw = [r for r in raw if r != 'NOT APPLICABLE']
    if not raw:
        return set()
    out = set()
    for stat in raw:
        if stat == 'GUILTY':
            out.add('PLED_GUILTY_TO_CHARGES')
        elif stat == 'NOT_GUILTY':
            out.add('PLED_NOT_GUILTY_TO_CHARGES')
    return out


def plea_status(proceeding):
    if not isinstance(proceeding, str):
        return 'NONE'
    output = list(extract_unique_plea_set(proceeding))
    if not output:
        return 'NONE'
    if 'PLED_GUILTY_TO_CHARGES' in output:
        return 'PLED_GUILTY_TO_CHARGES'
    elif 'PLED_NOT_GUILTY_TO_CHARGES' in output:
        return 'PLED_NOT_GUILTY_TO_CHARGES'
    else:
        return 'NONE'
    
# SECTION: Verdict Status

## Pattern
guilt_or_not_guilt_found_pattern = r'(?:DEFENDANT\s+)?(?:FOUND|DETERMINED\s+TO\s+BE|WAS)\s+(?:NOT\s+)?GUILTY(?:\s+AS\s+CHARGED)?|(?:VERDICT|COURT\s+FOUND).*?(?:NOT\s+)?GUILTY(?:\s+AS\s+CHARGED)?|(?:GUILTY|NOT\s+GUILTY)\s+OF\s+(?:RS|[A-Z])' 
GUILT_OR_NO_GUILT_REGEX = re.compile(guilt_or_not_guilt_found_pattern, re.VERBOSE | re.IGNORECASE)

## Extractor

def extract_verdict_status(text):
    if not isinstance(text, str):
        return None
    m = GUILT_OR_NO_GUILT_REGEX.findall(text)
    results = []
    for match in m:
        # Convert to uppercase for consistent matching
        match_upper = match.upper()
        # split match_upper and rejoin it
        match_upper = ' '.join(match_upper.split())
        if 'NOT GUILTY' in match_upper:
            results.append('NOT_GUILTY')
        elif 'GUILTY' in match_upper:
            results.append('GUILTY')
        else:
            results.append('NOT APPLICABLE')
    return results

def extract_unique_verdict_set(text: str) -> set[str]:
    raw = extract_verdict_status(text) or []
    # drop N/A and map
    raw = [r for r in raw if r != 'NOT APPLICABLE']
    if not raw:
        return set()
    out = set()
    for stat in raw:
        if stat == 'GUILTY':
            out.add('FOUND_GUILTY_BY_JUDGE_OR_JURY')
        elif stat == 'NOT_GUILTY':
            out.add('FOUND_NOT_GUILTY_BY_JUDGE_OR_JURY')
    return out

def guilty_or_no_guilty(proceeding):
    if not isinstance(proceeding, str):
        return 'NONE'
    output = list(extract_unique_verdict_set(proceeding))
    if not output:
        return 'NONE'
    if 'FOUND_GUILTY_BY_JUDGE_OR_JURY' in output:
        return 'FOUND_GUILTY_BY_JUDGE_OR_JURY'
    elif 'FOUND_NOT_GUILTY_BY_JUDGE_OR_JURY' in output:
        return 'FOUND_NOT_GUILTY_BY_JUDGE_OR_JURY'
    else:
        return 'NONE'

def verdict_status(proceeding):
    if not isinstance(proceeding, str):
        return 'NONE'
    proceeding_verdict = guilty_or_no_guilty(proceeding)
    if proceeding_verdict == 'FOUND_GUILTY_BY_JUDGE_OR_JURY':
        return 'TRIAL_VERDICT_REACHED'
    elif proceeding_verdict == 'FOUND_NOT_GUILTY_BY_JUDGE_OR_JURY':
        return 'TRIAL_VERDICT_REACHED'
    else:
        return 'NONE'
    
# SECTION: Sentencing Status

deferred_suspended_sentence_pattern = r"""
(?:
    (?:THE\s*)?COURT\s*(?:DEFERRED|SUSPENDED)\s*THE\s*FOLLOWING\s*SENTENCE\s*
    (?:\,?\s*)?
    UNDER\s*
    (?:ARTICLE\s*)?
    (893|894)
    |
    # Imposition of sentence suspended under Article 893 or 894
    IMPOSITION\s*OF\s*SENTENCE\s*(?:SUSPENDED|DEFERRED)\s*
    (?:\,?\s*)?
    UNDER\s*
    (?:ARTICLE\s*)?
    (893|894)
)
"""

compiled_deferred_suspended_pattern = re.compile(
    deferred_suspended_sentence_pattern, 
    flags=re.VERBOSE | re.MULTILINE
)


sentence_pattern = re.compile(r'(?:SENTENCE:|THE\s*SENTENCE\s*IS)((?:(?!SENTENCE:|THE\s*SENTENCE\s*IS)[\s\S])*)', re.MULTILINE)

## Extractor

def recieved_sentence(proceeding):
  proceedings = '\n'.join(line.strip() for line in proceeding.splitlines())
  if re.search(sentence_pattern, proceedings):
    return 'Yes'
  else:
    return 'No'

def find_facilities(text):

    if pd.isna(text):
        return []
    text = str(text)

    # Exclusions (unchanged)
    exclude_suffixes = [
        'INMATE', "SELF-HELP PROGRAMS", "SELF HELP PROGRAMS", 'WORK RELEASE PROGRAM',
        'SENTENCE IS COMPLETED', 'SHOULD TAKE NOTICE', "SELF PROGRAMS", 'ABOUT FAC'
    ]
    exclude_prefixes = [
        'PROGRAMS WITHIN',
        'SELF HELP PROGRAMS WITHIN',
        'RETURNED TO',
        'FOR TIME SERVED AT',
        'PROGRAM WITHIN',
        'SELF HELP PROGRAMS WHILE IN',
        'REMANDED TO CUSTODY OF',
        'RECOMMENDED FOR'
    ]

    # Keep your two views (unchanged)
    text_multiline = text.upper()
    text_single_line = ' '.join(text.split('\n')).upper()

    # Helper (unchanged semantics)
    def is_valid_occurrence(view: str, start: int, end: int) -> bool:
        # suffix window
        suffix = view[end:end+30]
        while suffix and suffix[0] in string.punctuation:
            suffix = suffix[1:]
        suffix = suffix.lstrip()
        for suf in exclude_suffixes:
            if suffix.startswith(suf):
                return False

        # prefix window
        prefix = view[max(0, start-30):start].rstrip()
        while prefix and prefix[-1] in string.punctuation:
            prefix = prefix[:-1]
        for pre in exclude_prefixes:
            if prefix.endswith(pre):
                return False
        return True

    # >>> CHANGE STARTS HERE: use regex instead of str.find <<<
    potential = []
    for pat in total_facilities_for_sentencing:               # facility_patterns remains a list
        rx = re.compile(pat, re.IGNORECASE)     # pat can use r'' and (?:...)
        for view in (text_multiline, text_single_line):
            for m in rx.finditer(view):         # regex search (not str.find)
                s, e = m.span()
                if is_valid_occurrence(view, s, e):
                    potential.append((m.group().upper(), s, e))
    # >>> CHANGE ENDS HERE <<<

    if not potential:
        return []

    # De-dupe and sort by start asc, length desc
    potential = list(set(potential))
    potential.sort(key=lambda x: (x[1], -(x[2]-x[1])))

    # Resolve overlaps by keeping the longest
    kept = []
    cur = potential[0]
    for nxt in potential[1:]:
        if nxt[1] < cur[2]:
            if (nxt[2]-nxt[1]) > (cur[2]-cur[1]):
                cur = nxt
        else:
            kept.append(cur)
            cur = nxt
    kept.append(cur)

    # Unique, preserve order
    out, seen = [], set()
    for val, _, _ in kept:
        if val not in seen:
            seen.add(val)
            out.append(val.strip())
    return out

def sentence_jail_status(proceeding):
    def facility_status(text):
        return find_facilities(text) if isinstance(text, str) else 'NONE'
    if compiled_deferred_suspended_pattern.search(proceeding):
        return "SENTENCE_THAT_DOES_NOT_INCLUDE_JAIL_TIME_WAS_IMPOSED"
    if recieved_sentence(proceeding) == 'Yes':
        return "SENTENCE_THAT_INCLUDES_JAIL_TIME_WAS_IMPOSED" if len(facility_status(proceeding)) > 0 else 'SENTENCE_THAT_DOES_NOT_INCLUDE_JAIL_TIME_WAS_IMPOSED'
    else:
        return 'NO_SENTENCE_WAS_IMPOSED'
# SECTION Example

## Patterns
ror_base_pattern = r'\b(?:R\.?\s*O\.?\s*R\.?\s*|RECOGNIZANCE)\b'
ror_antipattern = r'(?:NOT\s*ELIGIBLE|PROCEDURE\s*PROHIBITS|DID\s*NOT\s*ISSUE|DENIED|NO\s*BOND\s*OR|RESCINDED)'

positive_ror_pattern = rf'''(?x)
    (?<!(?:{ror_antipattern})\s+)
    \b(?:
        # \bR\.?\s*O\.?\s*R\.?\s*\b Bond Posted/Prepared
        \bR\.?\s*O\.?\s*R\.?\s*\b\s+BOND\s+(?:POSTED|WAS\s+PREPARED)|
        \$\d+(?:,\d{3})*(?:\.\d{2})?\s+\bR\.?\s*O\.?\s*R\.?\s*\b\s+BOND|
        \bR\.?\s*O\.?\s*R\.?\s*\b\s+BOND\s+\#\d+\s+WAS\s+PREPARED|
        \bR\.?\s*O\.?\s*R\.?\s*\b\s+BOND\s+IN\s+THE\s+AMOUNT\s+OF\s+\$\d+(?:,\d{3})*(?:\.\d{2})?\s+(?:WAS\s+)?POSTED|
        
        # Will Issue \bR\.?\s*O\.?\s*R\.?\s*\b
        WILL\s+ISSUE\s+(?:A|AN)?\s*\bR\.?\s*O\.?\s*R\.?\s*\b(?:\s+FOR\s+THIS\s+DEFENDANT)?|
        
        # Released on Own Recognizance
        WAS\s+RELEASED\s+ON\s+(?:HIS|HER|THEIR|OWN)\s+(?:OWN\s+)?RECOGNIZANCE|
        
        # Court Actions - Released/Set/Issued/Granted
        COURT\s+(?:
            RELEASED\s+THE\s+DEFENDANT\s+ON\s+(?:A\s+)?\$\d+(?:,\d{3})*(?:\.\d{2})?\s+\bR\.?\s*O\.?\s*R\.?\s*\b\s+BOND|
            SET\s+(?:A|AN)?\s*\bR\.?\s*O\.?\s*R\.?\s*\b\s+BOND(?:\s+IN\s+THE\s+AMOUNT\s+OF)?|
            ISSUED\s+(?:A|AN)?\s*\bR\.?\s*O\.?\s*R\.?\s*\b|
            GRANTED\s+(?:A|AN)?\s*\bR\.?\s*O\.?\s*R\.?\s*\b|
            GRANTED\s+(?:A\s+)?\$\d+(?:,\d{3})*(?:\.\d{2})?\s+\bR\.?\s*O\.?\s*R\.?\s*\b(?:\s+FOR)?|
            ALLOWED\s+DEFENDANT\s+(?:TO\s+BE\s+)?RELEASED\s+ON\s+\bR\.?\s*O\.?\s*R\.?\s*\b|
            GRANTED\s+THE\s+DEFENSE(?:'S|S)?\s+(?:ORAL\s+)?MOTION\s+TO\s+RELEASE\s+THE\s+DEFENDANT\s+ON\s+(?:A|AN)?\s*\bR\.?\s*O\.?\s*R\.?\s*\b|
            ORDERED\s+THE\s+DEFENDANT(?:'S|S)?\s+\bR\.?\s*O\.?\s*R\.?\s*\b|
            DID\s+GRANT\s+THE\s+DEFENDANT\s+(?:A|AN)?\s*\bR\.?\s*O\.?\s*R\.?\s*\b
        )|
        
        # \bR\.?\s*O\.?\s*R\.?\s*\b Issued Patterns
        \bR\.?\s*O\.?\s*R\.?\s*\b\s+ISSUED(?:\s+FOR\s+THIS\s+DEFENDANT)?|
        \bR\.?\s*O\.?\s*R\.?\s*\b\s+(?:WAS\s+)?ISSUED|
        \bR\.?\s*O\.?\s*R\.?\s*\b\s+BOND\s+(?:WAS\s+)?ISSUED|
        
        # Defendant Released Patterns
        DEFENDANT\s+(?:WAS\s+)?RELEASED\s+ON\s+(?:
            \bR\.?\s*O\.?\s*R\.?\s*\b|
            (?:HIS|HER|THEIR|OWN)\s+(?:OWN\s+)?RECOGNIZANCE
        )|
        
        # Gave Defendant \bR\.?\s*O\.?\s*R\.?\s*\b
        GAVE\s+DEFENDANT\s+(?:A\s+)?\$\d+(?:,\d{3})*(?:\.\d{2})?\s+\bR\.?\s*O\.?\s*R\.?\s*\b\s+BOND|
        
        # \bR\.?\s*O\.?\s*R\.?\s*\b by Commissioner/Judge
        \bR\.?\s*O\.?\s*R\.?\s*\b\s+(?:BY\s+)?(?:BLACKBURN|FRIEDMAN)|
        
        # Received \bR\.?\s*O\.?\s*R\.?\s*\b
        RECEIVED\s+(?:A|AN)?\s*\bR\.?\s*O\.?\s*R\.?\s*\b(?:\s+BOND)?\s+FROM|
        
        \bR\.?\s*O\.?\s*R\.?\s*\b\s+BOND(?:\s+\#\d+)
    )\b
'''
ror_base_regex         = regex.compile(ror_base_pattern, regex.IGNORECASE)
ror_antipattern_regex  = regex.compile(ror_antipattern,   regex.IGNORECASE)
positive_ror_regex = regex.compile(positive_ror_pattern, regex.IGNORECASE)

## Extractor

def should_include_ror(text):
    if pd.isna(text):
        return 'NONE'
    txt = str(text).upper()
    # 1) must match the base pattern
    if not ror_base_regex.search(txt):
        return 'NONE'
    # 2) if it matches an antipattern, only include if we also find a positive-ROR override
    if ror_antipattern_regex.search(txt):
        return 'DEFENDANT_RELEASED_ON_OWN_RECOGNIZANCE' if positive_ror_regex.search(txt) else 'NONE'
    # 3) otherwise (no antipattern), include any record that passed the base check
    return 'DEFENDANT_RELEASED_ON_OWN_RECOGNIZANCE'
