'use client'

import { useEffect, useRef } from 'react'

interface CodeSnippet {
  id: number
  x: number
  y: number
  vx: number
  vy: number
  baseVx?: number // Original velocity direction for baseline speed
  baseVy?: number // Original velocity direction for baseline speed  
  text: string
  size: number
  opacity: number
}

interface ExclusionZone {
  centerX: number
  centerY: number
  radius: number
}

export default function FloatingCodeBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const snippetsRef = useRef<CodeSnippet[]>([])
  const mouseRef = useRef({ x: 0, y: 0 })
  const animationRef = useRef<number>()
  const exclusionZoneRef = useRef<ExclusionZone>({ centerX: 0, centerY: 0, radius: 0 })

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

    // Set canvas size to cover full document
    const resizeCanvas = () => {
      const docHeight = Math.max(
        document.body.scrollHeight,
        document.documentElement.scrollHeight,
        window.innerHeight
      )
      canvas.width = window.innerWidth
      canvas.height = docHeight
    }
    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)
    
    // Also resize when content changes (use MutationObserver)
    const observer = new MutationObserver(resizeCanvas)
    observer.observe(document.body, { childList: true, subtree: true })

    // Function to update exclusion zone based on profile picture position
    const updateExclusionZone = () => {
      const profilePic = document.querySelector('img[alt="Profile Picture"]')
      if (profilePic) {
        const rect = profilePic.getBoundingClientRect()
        const scrollY = window.scrollY
        // Calculate center of the circular profile picture
        exclusionZoneRef.current = {
          centerX: rect.left + rect.width / 2,
          centerY: rect.top + scrollY + rect.height / 2,
          radius: Math.max(rect.width, rect.height) / 2 + 20 // Add padding
        }
      }
    }
    
    // Update exclusion zone on resize and scroll
    updateExclusionZone()
    window.addEventListener('resize', updateExclusionZone)
    window.addEventListener('scroll', updateExclusionZone)

    // Initialize code snippets - more snippets to cover full document height
    const initSnippets = () => {
      snippetsRef.current = []
      // Scale snippet count based on document height
      const snippetCount = Math.max(48, Math.floor((canvas.height / window.innerHeight) * 32))
      for (let i = 0; i < snippetCount; i++) {
        snippetsRef.current.push({
          id: i,
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          // Initial speed of the code snippet
          vx: (Math.random() - 0.5) * 1,
          vy: (Math.random() - 0.5) * 1,
          text: codeTexts[Math.floor(Math.random() * codeTexts.length)],
          size: Math.random() * 8 + 10,
          opacity: Math.random() * 0.3 + 0.3
        })
      }
    }
    initSnippets()

    // Mouse tracking - account for scroll position
    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY + window.scrollY }
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

        // Check if snippet is in exclusion zone (circular profile photo area)
        const zone = exclusionZoneRef.current
        const distToCenter = Math.sqrt(
          Math.pow(snippet.x - zone.centerX, 2) + 
          Math.pow(snippet.y - zone.centerY, 2)
        )
        const inExclusionZone = distToCenter < zone.radius

        // Draw snippet only if not in exclusion zone
        if (!inExclusionZone) {
          ctx.font = `${snippet.size}px 'Courier New', monospace`
          ctx.fillStyle = `rgba(59, 130, 246, ${snippet.opacity})` // Blue color
          ctx.fillText(snippet.text, snippet.x, snippet.y)
        }
      })

      animationRef.current = requestAnimationFrame(animate)
    }
    animate()

    return () => {
      window.removeEventListener('resize', resizeCanvas)
      window.removeEventListener('resize', updateExclusionZone)
      window.removeEventListener('scroll', updateExclusionZone)
      window.removeEventListener('mousemove', handleMouseMove)
      observer.disconnect()
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="absolute top-0 left-0 w-full pointer-events-none z-0"
      style={{ background: 'transparent' }}
    />
  )
}
