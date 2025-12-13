-- ============================================================================
-- DATA WAREHOUSE - HISTORIAS CLÍNICAS SPACCC (SIMPLIFICADO)
-- DDL para SQL Server
-- ============================================================================

-- Crear base de datos
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'DW_HistoriasClinicas')
BEGIN
    CREATE DATABASE DW_HistoriasClinicas;
END
GO

USE DW_HistoriasClinicas;
GO

-- ============================================================================
-- TABLA STAGING (carga desde CSV via SSIS)
-- ============================================================================

CREATE TABLE staging_historias (
    nombre_archivo      NVARCHAR(255),
    texto_original      NVARCHAR(MAX),
    especialidad        NVARCHAR(100),
    diagnostico         NVARCHAR(100),
    tratamiento         NVARCHAR(100),
    edad_paciente       INT,
    sexo_paciente       NVARCHAR(20),
    confidence_score    DECIMAL(5,4),
    fecha_procesamiento DATETIME
);
GO

-- ============================================================================
-- DIMENSIONES
-- ============================================================================

CREATE TABLE dim_especialidad (
    id_especialidad     INT IDENTITY(1,1) PRIMARY KEY,
    nombre              NVARCHAR(100) NOT NULL UNIQUE
);
GO

CREATE TABLE dim_diagnostico (
    id_diagnostico      INT IDENTITY(1,1) PRIMARY KEY,
    nombre              NVARCHAR(100) NOT NULL UNIQUE
);
GO

CREATE TABLE dim_tratamiento (
    id_tratamiento      INT IDENTITY(1,1) PRIMARY KEY,
    nombre              NVARCHAR(100) NOT NULL UNIQUE
);
GO

-- ============================================================================
-- TABLA DE HECHOS
-- ============================================================================

CREATE TABLE fact_historia_clinica (
    id_historia         INT IDENTITY(1,1) PRIMARY KEY,
    id_especialidad     INT NOT NULL,
    id_diagnostico      INT NOT NULL,
    id_tratamiento      INT NOT NULL,
    nombre_archivo      NVARCHAR(255) NOT NULL,
    texto_original      NVARCHAR(MAX),
    edad_paciente       INT,
    sexo_paciente       NVARCHAR(20),
    confidence_score    DECIMAL(5,4),
    fecha_procesamiento DATETIME,
    CONSTRAINT FK_especialidad FOREIGN KEY (id_especialidad) REFERENCES dim_especialidad(id_especialidad),
    CONSTRAINT FK_diagnostico FOREIGN KEY (id_diagnostico) REFERENCES dim_diagnostico(id_diagnostico),
    CONSTRAINT FK_tratamiento FOREIGN KEY (id_tratamiento) REFERENCES dim_tratamiento(id_tratamiento)
);
GO
