package com.example.ner.controller;

import com.example.ner.dto.NerRequest;
import com.example.ner.dto.NerResponse;
import com.example.ner.service.NerService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/ner")
@CrossOrigin(origins = "*")
public class NerController {

    private final NerService nerService;

    // l'injection des dépendances
    public NerController(NerService nerService) {
        this.nerService = nerService;
    }

    @PostMapping("/analyze")
    public ResponseEntity<NerResponse> analyzeText(@RequestBody NerRequest request) {
        NerResponse reponse = nerService.analyzeText(request.text());
        return ResponseEntity.ok(reponse);
    }
}