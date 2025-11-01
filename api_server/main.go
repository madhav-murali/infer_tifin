package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"

	"github.com/gin-gonic/gin"
)

const HF_SPACE_URL = "https://trinitysoul-infer-tifin.hf.space/infer"

type ChatRequest struct {
	ChatID       string `json:"chat_id"`
	SystemPrompt string `json:"system_prompt"`
	UserPrompt   string `json:"user_prompt"`
}

type BatchRequest struct {
	Queries []ChatRequest `json:"queries"`
}

type ModelResponse struct {
	Response string `json:"response"`
}

func callModelAPI(req ChatRequest) (string, error) {
	body, _ := json.Marshal(req)
	resp, err := http.Post(HF_SPACE_URL, "application/json", bytes.NewBuffer(body))
	if err != nil {
		fmt.Printf("Error: %s", err)
		return "", err
	}

	defer resp.Body.Close()

	data, _ := io.ReadAll(resp.Body)
	var modelResp ModelResponse
	json.Unmarshal(data, &modelResp)
	return modelResp.Response, nil

}

func main() {
	r := gin.Default()

	r.POST("/chat", func(c *gin.Context) {
		var req ChatRequest
		if err := c.BindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		resp, err := callModelAPI(req)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"response": resp})
	})

	r.POST("/chat/batched", func(c *gin.Context) {
		var batchReq BatchRequest
		if err := c.BindJSON(&batchReq); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		var wg sync.WaitGroup
		responses := make([]string, len(batchReq.Queries))

		for i, q := range batchReq.Queries {
			wg.Add(1)
			go func(i int, q ChatRequest) {
				defer wg.Done()
				resp, err := callModelAPI(q)
				if err != nil {
					responses[i] = "Error: " + err.Error()
				} else {
					responses[i] = resp
				}
			}(i, q)
		}

		wg.Wait()
		c.JSON(http.StatusOK, gin.H{"responses": responses})
	})

	r.Run(":8080")
}
