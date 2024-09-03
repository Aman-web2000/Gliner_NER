from gliner import GLiNER
import streamlit as st
from dict import labels


def ner_model(text,model_version):
    try:
        model =GLiNER.from_pretrained(model_version)

        # text="""Hello this is Aman Chauhan, I'm 24 years old and I was born in Rishikesh, Uttarakhand. I'm currently working deloitte USI"""

        # labels=["Person", "Award" ,"Age", "Date", "Organization", "Location"]

        # Sample text for entity prediction
        # text = """
        # Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
        # """

        # Labels for entity prediction
        # Most GLiNER models should work best when entity types are in lower case or title case
        

        entities=model.predict_entities(text, labels, threshold=0.5)

        return entities

    except Exception as e:
        print(f"Error : {e}")

