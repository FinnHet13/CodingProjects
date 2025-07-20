import Image from "next/image"
import Link from "next/link"
import { ExternalLink, Github, Mail, MapPin } from "lucide-react"

export default function HomePage() {
  const projects = [
    {
      title: "E-Commerce Platform",
      description:
        "A full-stack e-commerce solution built with Next.js, TypeScript, and Stripe integration. Features include user authentication, product management, and secure payment processing.",
      technologies: ["Next.js", "TypeScript", "Stripe", "Prisma"],
      githubUrl: "https://github.com/username/ecommerce-platform",
      liveUrl: "https://ecommerce-demo.vercel.app",
    },
    {
      title: "Task Management App",
      description:
        "A collaborative task management application with real-time updates, drag-and-drop functionality, and team collaboration features.",
      technologies: ["React", "Node.js", "Socket.io", "MongoDB"],
      githubUrl: "https://github.com/username/task-manager",
      liveUrl: "https://taskmanager-demo.vercel.app",
    },
    {
      title: "Weather Dashboard",
      description:
        "A responsive weather dashboard that displays current conditions and forecasts using OpenWeatherMap API with interactive charts and location-based services.",
      technologies: ["Vue.js", "Chart.js", "OpenWeatherMap API", "Tailwind CSS"],
      githubUrl: "https://github.com/username/weather-dashboard",
      liveUrl: "https://weather-dashboard-demo.vercel.app",
    },
    {
      title: "Machine Learning Model Deployment",
      description:
        "A Flask web application that serves a trained machine learning model for image classification with a clean REST API interface.",
      technologies: ["Python", "Flask", "TensorFlow", "Docker"],
      githubUrl: "https://github.com/username/ml-deployment",
      liveUrl: null,
    },
  ]

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header/Profile Section */}
      <header className="border-b border-gray-800">
        <div className="max-w-4xl mx-auto px-6 py-12">
          <div className="flex flex-col md:flex-row items-start gap-8">
            <div className="flex-shrink-0">
              <Image
                src="/assets/profile-pic.jpg"
                alt="Profile Picture"
                width={200}
                height={200}
                className="rounded-full border-2 border-gray-700"
              />
            </div>
            <div className="flex-1">
              <h1 className="text-4xl font-bold mb-4">John Doe</h1>
              <p className="text-xl text-gray-300 mb-6">Full Stack Developer & Software Engineer</p>
              <p className="text-gray-400 mb-6 leading-relaxed">
                Passionate about creating efficient, scalable solutions and exploring new technologies. I enjoy working
                on both frontend and backend development, with a focus on modern web technologies and clean,
                maintainable code.
              </p>
              <div className="flex flex-wrap gap-4 text-sm">
                <div className="flex items-center gap-2 text-gray-400">
                  <MapPin className="w-4 h-4" />
                  San Francisco, CA
                </div>
                <Link
                  href="mailto:john.doe@example.com"
                  className="flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors"
                >
                  <Mail className="w-4 h-4" />
                  john.doe@example.com
                </Link>
                <Link
                  href="https://github.com/johndoe"
                  className="flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors"
                >
                  <Github className="w-4 h-4" />
                  GitHub
                </Link>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Projects Section */}
      <main className="max-w-4xl mx-auto px-6 py-12">
        <h2 className="text-3xl font-bold mb-8">Projects</h2>
        <div className="space-y-8">
          {projects.map((project, index) => (
            <article
              key={index}
              className="border border-gray-800 rounded-lg p-6 hover:border-gray-700 transition-colors"
            >
              <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4 mb-4">
                <h3 className="text-xl font-semibold text-blue-400">{project.title}</h3>
                <div className="flex gap-3">
                  <Link
                    href={project.githubUrl}
                    className="flex items-center gap-1 text-sm text-gray-400 hover:text-white transition-colors"
                  >
                    <Github className="w-4 h-4" />
                    Code
                  </Link>
                  {project.liveUrl && (
                    <Link
                      href={project.liveUrl}
                      className="flex items-center gap-1 text-sm text-gray-400 hover:text-white transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      Live Demo
                    </Link>
                  )}
                </div>
              </div>
              <p className="text-gray-300 mb-4 leading-relaxed">{project.description}</p>
              <div className="flex flex-wrap gap-2">
                {project.technologies.map((tech, techIndex) => (
                  <span
                    key={techIndex}
                    className="px-3 py-1 text-xs bg-gray-800 text-gray-300 rounded-full border border-gray-700"
                  >
                    {tech}
                  </span>
                ))}
              </div>
            </article>
          ))}
        </div>

        {/* Footer */}
        <footer className="mt-16 pt-8 border-t border-gray-800 text-center text-gray-500">
          <p>© 2024 John Doe. Built with Next.js and deployed on Vercel.</p>
        </footer>
      </main>
    </div>
  )
}
