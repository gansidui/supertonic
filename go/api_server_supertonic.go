package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/gansidui/simplelogger"
	"github.com/gin-gonic/gin"
	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	ort "github.com/yalue/onnxruntime_go"
)

// Server config constants
const (
	Port            = 8000
	OnnxDir         = "assets/onnx"
	VoiceStyleDir   = "assets/voice_styles"
	TotalStep       = 5
	Speed           = 1.0
	SilenceDuration = 0.3
	TTSPoolSize     = 2
)

// Global variables
var (
	cfg        Config
	ttsPool    chan *TextToSpeech        // model doesn't support concurrent inference, so we need to use a pool to manage the models
	styleCache = make(map[string]*Style) // loaded at init, read-only after
)

// TTSRequest holds TTS request parameters
type TTSRequest struct {
	SpeakerName string  `json:"speaker_name" form:"speaker_name"`
	Text        string  `json:"text" form:"text"`
	Lang        string  `json:"lang" form:"lang"`
	VolumeGain  float32 `json:"volume_gain" form:"volume_gain"` // only applies when > 1.0
}

func main() {
	// Setup logger
	slogger := &simplelogger.Logger{MaxSize: 100}
	slogger.Open(fmt.Sprintf("log_%d.txt", Port))
	log.SetOutput(slogger)

	// Initialize ONNX Runtime
	log.Println("Initializing ONNX Runtime...")
	if err := InitializeONNXRuntime(); err != nil {
		log.Fatalf("Failed to initialize ONNX Runtime: %v", err)
	}
	defer ort.DestroyEnvironment()

	// Load config
	log.Println("Loading TTS config...")
	var err error
	cfg, err = LoadCfgs(OnnxDir)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Initialize TTS instance pool
	log.Printf("Loading TTS model pool (size=%d)...", TTSPoolSize)
	ttsPool = make(chan *TextToSpeech, TTSPoolSize)
	for i := 0; i < TTSPoolSize; i++ {
		tts, err := LoadTextToSpeech(OnnxDir, false, cfg)
		if err != nil {
			log.Fatalf("Failed to load TTS model instance %d: %v", i, err)
		}
		ttsPool <- tts
		log.Printf("Loaded TTS instance %d", i)
	}

	// Preload all voice styles
	log.Println("Preloading voice styles...")
	if err := preloadVoiceStyles(); err != nil {
		log.Fatalf("Failed to preload some voice styles: %v", err)
	}

	// Setup gin router
	gin.SetMode(gin.ReleaseMode)
	router := gin.Default()

	router.GET("/", homeHandler)
	router.POST("/", homeHandler)
	router.GET("/tts", ttsHandler)
	router.POST("/tts", ttsHandler)

	// Start server
	addr := fmt.Sprintf("0.0.0.0:%d", Port)
	log.Printf("Starting server, listening on %s", addr)
	log.Printf("Available languages: %v", AvailableLangs)
	log.Printf("Voice styles directory: %s", VoiceStyleDir)

	server := &http.Server{
		Addr:         addr,
		Handler:      router,
		ReadTimeout:  600 * time.Second,
		WriteTimeout: 600 * time.Second,
	}

	// Start server in goroutine
	go func() {
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server failed: %v", err)
		}
	}()

	// Wait for interrupt signal to gracefully shutdown the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt, syscall.SIGTERM, syscall.SIGINT)
	sig := <-quit
	log.Printf("Received signal: %v, shutting down server...", sig)

	// Graceful shutdown with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		log.Printf("Server shutdown error: %v", err)
	}

	log.Println("Server exited")
}

// preloadVoiceStyles loads all voice styles from directory into cache
func preloadVoiceStyles() error {
	files, err := os.ReadDir(VoiceStyleDir)
	if err != nil {
		return fmt.Errorf("failed to read voice style directory: %w", err)
	}

	for _, file := range files {
		if file.IsDir() || filepath.Ext(file.Name()) != ".json" {
			continue
		}
		speakerName := file.Name()[:len(file.Name())-5] // remove .json
		voiceStylePath := filepath.Join(VoiceStyleDir, file.Name())
		style, err := LoadVoiceStyle([]string{voiceStylePath}, false)
		if err != nil {
			log.Printf("Warning: failed to load voice style %s: %v", speakerName, err)
			continue
		}
		styleCache[speakerName] = style
	}

	log.Printf("Loaded %d voice styles into cache", len(styleCache))
	return nil
}

// getVoiceStyle gets voice style from cache
func getVoiceStyle(speakerName string) (*Style, error) {
	if style, ok := styleCache[speakerName]; ok {
		return style, nil
	}
	return nil, fmt.Errorf("voice style not found: %s", speakerName)
}

func homeHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

func ttsHandler(c *gin.Context) {
	var req TTSRequest
	if err := c.ShouldBind(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Validate required parameters
	if req.SpeakerName == "" || req.Text == "" || req.Lang == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Missing required parameter: speaker_name or text or lang"})
		return
	}

	// Validate language
	if !isValidLang(req.Lang) {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("Invalid language: %s. Available: %v", req.Lang, AvailableLangs)})
		return
	}

	log.Printf("Received TTS request, speaker=%s, lang=%s, text_size=%d, volume_gain=%f",
		req.SpeakerName, req.Lang, len(req.Text), req.VolumeGain)

	// Get voice style from cache
	style, err := getVoiceStyle(req.SpeakerName)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("Invalid speaker_name: %s (%v)", req.SpeakerName, err)})
		return
	}

	// Get TTS instance from pool
	tts := <-ttsPool
	start := time.Now()
	wavData, duration, err := tts.Call(req.Text, req.Lang, style, TotalStep, Speed, SilenceDuration)
	if req.VolumeGain > 1.0 {
		wavData = applyGain(wavData, req.VolumeGain)
	}
	elapsed := time.Since(start)
	ttsPool <- tts // Return to pool

	if err != nil {
		log.Printf("TTS failed: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("TTS failed: %v", err)})
		return
	}

	var rtf float32
	if duration > 0 {
		rtf = float32(elapsed.Seconds()) / duration
	}
	log.Printf("TTS succeeded, speaker=%s, lang=%s, text_size=%d, duration=%.2fs, elapsed=%.2fs, rtf=%.2f",
		req.SpeakerName, req.Lang, len(req.Text), duration, elapsed.Seconds(), rtf)

	// Write WAV directly to response
	wavByts, err := encodeWav(wavData, tts.SampleRate)
	if err != nil {
		log.Printf("Failed to encode WAV: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to encode WAV: %v", err)})
		return
	}

	c.Header("Content-Disposition", `attachment; filename="output.wav"`)
	c.Data(http.StatusOK, "audio/wav", wavByts)
}

// memWriteSeeker is an in-memory io.WriteSeeker
type memWriteSeeker struct {
	buf []byte
	pos int
}

func (m *memWriteSeeker) Write(p []byte) (n int, err error) {
	minCap := m.pos + len(p)
	if minCap > cap(m.buf) {
		newBuf := make([]byte, minCap, minCap*2)
		copy(newBuf, m.buf)
		m.buf = newBuf
	}
	if minCap > len(m.buf) {
		m.buf = m.buf[:minCap]
	}
	copy(m.buf[m.pos:], p)
	m.pos += len(p)
	return len(p), nil
}

func (m *memWriteSeeker) Seek(offset int64, whence int) (int64, error) {
	var newPos int
	switch whence {
	case 0: // io.SeekStart
		newPos = int(offset)
	case 1: // io.SeekCurrent
		newPos = m.pos + int(offset)
	case 2: // io.SeekEnd
		newPos = len(m.buf) + int(offset)
	}
	if newPos < 0 {
		newPos = 0
	}
	m.pos = newPos
	return int64(m.pos), nil
}

func (m *memWriteSeeker) Bytes() []byte {
	return m.buf
}

// encodeWav encodes float32 audio data to WAV format bytes
func encodeWav(audioData []float32, sampleRate int) ([]byte, error) {
	// Convert float32 to int
	intData := make([]int, len(audioData))
	for i, sample := range audioData {
		clamped := math.Max(-1.0, math.Min(1.0, float64(sample)))
		intData[i] = int(clamped * 32767)
	}

	// Encode to WAV
	buf := &memWriteSeeker{buf: make([]byte, 0, len(audioData)*2+44)}
	encoder := wav.NewEncoder(buf, sampleRate, 16, 1, 1)
	audioBuf := &audio.IntBuffer{
		Data:           intData,
		Format:         &audio.Format{SampleRate: sampleRate, NumChannels: 1},
		SourceBitDepth: 16,
	}

	if err := encoder.Write(audioBuf); err != nil {
		return nil, err
	}
	if err := encoder.Close(); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func applyGain(samples []float32, gain float32) []float32 {
	for i, s := range samples {
		v := s * gain
		if v > 1.0 {
			v = 1.0
		} else if v < -1.0 {
			v = -1.0
		}
		samples[i] = v
	}
	return samples
}
