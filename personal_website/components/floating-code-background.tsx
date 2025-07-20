'use client'

import { useEffect, useRef } from 'react'

interface CodeSnippet {
  id: number
  x: number
  y: number
  vx: number
  vy: number
  text: string
  size: number
  opacity: number
}

export default function FloatingCodeBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const snippetsRef = useRef<CodeSnippet[]>([])
  const mouseRef = useRef({ x: 0, y: 0 })
  const animationRef = useRef<number>()

  // Sample code snippets to display
  const codeTexts = [
    'const', 'function', 'return', '=>', '{}', '[]', '<>', 'import', 'export',
    'async', 'await', 'class', 'let', 'var', 'if', 'else', 'for', 'while',
    '&&', '||', '==', '!==', '...', 'map()', 'filter()', 'reduce()',
    'useState', 'useEffect', 'props', 'state', 'render', 'component'
  ]

  // Constants for the animation
  const BASE_SPEED = 0.3 // Baseline speed snippets return to
  const SPEED_RETURN_FORCE = 0.01 // How quickly snippets return to baseline

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)

    // Initialize code snippets
    const initSnippets = () => {
      snippetsRef.current = []
      for (let i = 0; i < 32; i++) {
        snippetsRef.current.push({
          id: i,
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          // Initial speed of the code snippet
          vx: (Math.random() - 0.5) * 1,
          vy: (Math.random() - 0.5) * 1,
          text: codeTexts[Math.floor(Math.random() * codeTexts.length)],
          size: Math.random() * 8 + 10,
          opacity: Math.random() * 0.3 + 0.1
        })
      }
    }
    initSnippets()

    // Mouse tracking
    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY }
    }
    window.addEventListener('mousemove', handleMouseMove)

    // Animation loop
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
  
      snippetsRef.current.forEach(snippet => {
      // Store original velocity direction for baseline speed
        if (!snippet.baseVx) {
          snippet.baseVx = snippet.vx > 0 ? BASE_SPEED : -BASE_SPEED
          snippet.baseVy = snippet.vy > 0 ? BASE_SPEED : -BASE_SPEED
        }

        // Mouse repulsion effect
        const dx = snippet.x - mouseRef.current.x
        const dy = snippet.y - mouseRef.current.y
        const distance = Math.sqrt(dx * dx + dy * dy)
        
        // If the mouse is close, apply a repulsion force
        if (distance < 100) {
          const force = (100 - distance) / 100
          snippet.vx += (dx / distance) * force * 0.02
          snippet.vy += (dy / distance) * force * 0.02
        }

        // Return to baseline speed when not affected by mouse
        const currentSpeed = Math.sqrt(snippet.vx * snippet.vx + snippet.vy * snippet.vy)

        if (distance > 100 && currentSpeed < BASE_SPEED) {
          // Gradually return to baseline speed
          snippet.vx += (snippet.baseVx - snippet.vx) * SPEED_RETURN_FORCE
          snippet.vy += (snippet.baseVy - snippet.vy) * SPEED_RETURN_FORCE
        } else if (distance > 100) {
          // Apply gentle friction only when moving faster than baseline
          if (currentSpeed > BASE_SPEED) {
            snippet.vx *= 0.99
            snippet.vy *= 0.99
            }
        }

        // Update position
        snippet.x += snippet.vx
        snippet.y += snippet.vy

        // Boundary wrapping
        if (snippet.x < -50) snippet.x = canvas.width + 50
        if (snippet.x > canvas.width + 50) snippet.x = -50
        if (snippet.y < -50) snippet.y = canvas.height + 50
        if (snippet.y > canvas.height + 50) snippet.y = -50

        // Draw snippet
        ctx.font = `${snippet.size}px 'Courier New', monospace`
        ctx.fillStyle = `rgba(59, 130, 246, ${snippet.opacity})` // Blue color
        ctx.fillText(snippet.text, snippet.x, snippet.y)
      })

      animationRef.current = requestAnimationFrame(animate)
    }
    animate()

    return () => {
      window.removeEventListener('resize', resizeCanvas)
      window.removeEventListener('mousemove', handleMouseMove)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none z-0"
      style={{ background: 'transparent' }}
    />
  )
}