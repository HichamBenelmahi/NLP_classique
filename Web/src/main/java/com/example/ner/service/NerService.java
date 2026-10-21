package com.example.ner.service;

import com.example.ner.dto.NerRequest;
import com.example.ner.dto.NerResponse;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

@Service
public class NerService {

    private final RestClient restClient;

    public NerService() {
        // Initialisation du client HTTP interne de Spring
        this.restClient = RestClient.create();
    }

    public NerResponse analyzeText(String text) {
        // Vérifications (Logique métier et sécurité)
        if (text == null || text.trim().isEmpty()) {
            throw new IllegalArgumentException("Le texte à analyser ne peut pas être vide.");
        }
        if (text.length() > 5000) {
            // Protection contre les textes trop longs qui épuiseraient la RAM du modèle
            throw new IllegalArgumentException("Le texte dépasse la limite autorisée par le modèle.");
        }

        //Appel HTTP vers l'API Python
        try {
            return restClient.post()
                    .uri("http://127.0.0.1:8000/predict")
                    .body(new NerRequest(text))
                    .retrieve()
                    .body(NerResponse.class);
        } catch (Exception e) {
            //erreur si le script Python (FastAPI) est éteint
            throw new RuntimeException("Le modèle biomédical est actuellement indisponible.");
        }
    }
}